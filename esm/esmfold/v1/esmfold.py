# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as T
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from omegaconf import MISSING
from openfold.data.data_transforms import make_atom14_masks
from openfold.np import residue_constants
from openfold.utils.loss import compute_predicted_aligned_error, compute_tm
from torch import nn
from torch.nn import LayerNorm

import esm
from esm import Alphabet
from esm.esmfold.v1.categorical_mixture import categorical_lddt
from esm.esmfold.v1.trunk import FoldingTrunk, FoldingTrunkConfig
from esm.esmfold.v1.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
    output_to_pdb,
)


@dataclass
class ESMFoldConfig:
    trunk: FoldingTrunkConfig = field(default_factory=FoldingTrunkConfig) #JO: Debug ESMFold, this is the characteristics of dataclass
    lddt_head_hid_dim: int = 128

 
class ESMFold(nn.Module):
    #JO: It can accept other parameters, but now actually only accept config from esmfold_3B_v1.pt
    def __init__(self, esmfold_config=None, **kwargs):
        super().__init__()

        self.cfg = esmfold_config if esmfold_config else ESMFoldConfig(**kwargs)
        cfg = self.cfg

        self.distogram_bins = 64
        #JO: they are esm2 model (include model shapes, weights and bias ...) and alphabet separately
        self.esm, self.esm_dict = esm.pretrained.esm2_t36_3B_UR50D()
        #JO: freeze the model
        self.esm.requires_grad_(False)
        #JO: half the model, float32 to float16
        self.esm.half()

        #JO: It really looks like the msa_feats. It's embeddings and has 'feats' in the name. The value is 2560
        self.esm_feats = self.esm.embed_dim
        print('ESM2 Embedding Dimention:\n',self.esm_feats)
        #JO: attention_heads is 40, num_layers is 36
        self.esm_attns = self.esm.num_layers * self.esm.attention_heads
        print('ESM2 Attention Heads:\n',self.esm.attention_heads)
        print('ESM2 Layers:\n',self.esm.num_layers)
        #JO: register a tensor as a buffer, it is not a parameter
        print("esm_dict (alphabet from esm2):", self.esm_dict)
        #JO: the reordered index can be referred to as af2_to_esm
        self.register_buffer("af2_to_esm", ESMFold._af2_to_esm(self.esm_dict))
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm.num_layers + 1))

        #JO: This is for folding trunk, sequence dimention is 1024, pairwise dimention is 128
        c_s = cfg.trunk.sequence_state_dim
        c_z = cfg.trunk.pairwise_state_dim
        print("folding trunk sequence_state_dim:\n",c_s)
        print("folding trunk pairwise_state_dim:\n",c_z)
        #JO: This is the Feed Neural Network for Transformer
        self.esm_s_mlp = nn.Sequential(
            #JO: Normalize over the last dimention and it should be the same as embedding dimention 2560
            #JO: Here the weights and bias are learnable
            #JO: size of esm_feats is C, and here is to transfer the representation to folding trunk size
            LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, c_s),
            #JO: ReLU is a non-linear activation function, suitable for complex relationships and also the 
            #JO: gradient is not vanishing
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

        #JO: Freeze the parameters of esm_s_mlp
        for param in self.esm_s_mlp.parameters():
            param.requires_grad = False

        #JO: 0 is padding, N - 1 is unknown residues, N is mask. n_tokens_embed is the number of indexes
        #JO: The left residues are already defined in function: _af2_to_esm()
        self.n_tokens_embed = residue_constants.restype_num + 3
        self.pad_idx = 0
        self.unk_idx = self.n_tokens_embed - 2
        self.mask_idx = self.n_tokens_embed - 1
        '''
        JO: So the sequence dimention is selected as the embedding size
        The weights are learnable and applys to the normalized distribution. It has the same dimention 
        as the embedding. padding_idx is to make sure the size of each input to embedding is the same, but 
        hope it will have no influence on the result, so it is set to 0 and not updated during training
        '''
        self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=0)
        #JO: folding trunk has triangular attention blocks inside, '**' is to discompose the dictionary
        self.trunk = FoldingTrunk(**cfg.trunk)
        '''
        JOJO: I would expect these layers to help analyze outputs from esm2
        '''
        self.distogram_head = nn.Linear(c_z, self.distogram_bins)
        self.ptm_head = nn.Linear(c_z, self.distogram_bins)
        self.lm_head = nn.Linear(c_s, self.n_tokens_embed)
        self.lddt_bins = 50
        self.lddt_head = nn.Sequential(
            nn.LayerNorm(cfg.trunk.structure_module.c_s),
            nn.Linear(cfg.trunk.structure_module.c_s, cfg.lddt_head_hid_dim),
            nn.Linear(cfg.lddt_head_hid_dim, cfg.lddt_head_hid_dim),
            nn.Linear(cfg.lddt_head_hid_dim, 37 * self.lddt_bins),
        )

    #JO: Static method has no relationship with the instance of the class. It is just a function inside the class
    @staticmethod
    #JO: attribute index to each token in esmfold according to the alphabet in esm2
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [
            d.get_idx(v) for v in residue_constants.restypes_with_x
        ]
        print("reordered esm:", esm_reorder)
        return torch.tensor(esm_reorder)

    def _af2_idx_to_esm_idx(self, aa, mask):
        #JO: This is where the masking takes effect, but it is not defined by me
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm[aa]

    def _compute_language_model_representations(
        self, esmaa: torch.Tensor
    ) -> torch.Tensor:
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0)

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        print("BOS and EOS index in esm2:", bosi, eosi)
        #JO: Size is (B, 1), and fill the tensor with bosi. So add bos token to the beginning of each sequence
        #JO: Add eos token to the end of each sequence
        bos = esmaa.new_full((batch_size, 1), bosi)
        #JO: Does this begin to add padding?
        eos = esmaa.new_full((batch_size, 1), self.esm_dict.padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi
        print("What esmaa looks like after adding bos, padding and eos:", esmaa)
        #JO: use the pretrained model to get the representation, this is the results after all the esm calculations
        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=False,
        )
        #JO: So res is a dictionary. 'logit' refers to the output logits (B, L, A)
        #JO: 'representations' refers to the output of each layer (nLayers, L, B, C)
        #JO: This step now is to stack the representations of each layer together and put them to dim=2
        #JO: So the final shape is (B, L, nLayers, C)
        esm_s = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
        #JO: Here only manipulate the second dimention, seems like to remove the bos and eos tokens
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
        return esm_s

    #JO: Change masked position to special index
    def _mask_inputs_to_esm(self, esmaa, pattern):
        new_esmaa = esmaa.clone()
        new_esmaa[pattern == 1] = self.esm_dict.mask_idx
        print("Mask_idx is: ", self.esm_dict.mask_idx)
        return new_esmaa

    #JO: aa here is the input sequence
    def forward(
        self,
        aa: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = None,
    ):
        """Runs a forward pass given input tokens. Use `model.infer` to
        run inference from a sequence.

        Args:
            aa (torch.Tensor): Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """

        if mask is None:
            mask = torch.ones_like(aa)
        '''
        #JO: So B is tthe number of batch, which is usually 1 for monomer, but considering that batch is usually used
        to adapt calculation to GPUs. I would take B here as the batch before chunking
        '''
        B = aa.shape[0] #JO: batch size
        L = aa.shape[1] #JO: sequence length
        device = aa.device

        if residx is None:
            residx = torch.arange(L, device=device).expand_as(aa)

        # === ESM ===
        esmaa = self._af2_idx_to_esm_idx(aa, mask)
        print("After adding padding idx and transfer amino acid to idx (Also make the idx start from 1):",esmaa)
        if masking_pattern is not None:
            #JO: make the index at the position of masking pattern to be the mask index, this is what we defined
            esmaa = self._mask_inputs_to_esm(esmaa, masking_pattern)
        print("After adding the masking patterns inside ",esmaa)
        #JO: get the representation of the sequence: B, L, nLayers, C
        esm_s = self._compute_language_model_representations(esmaa)
        #JO: shape of esm_s is (B, L, nLayers, C), with bos and eos removed
        #JO: Output of ADK1, [1, 214, 37, 2560]
        print("Shape of the result of ESM2 calculation",esm_s.shape)
        # Convert esm_s to the precision used by the trunk and
        # the structure module. These tensors may be a lower precision if, for example,
        # we're running the language model in fp16 precision.
        esm_s = esm_s.to(self.esm_s_combine.dtype)
        #JO: In order to make loss function work, I need to comment this step
        esm_s = esm_s.detach()

        # === preprocessing ===
        #JO: Here the esm_s_combine is all 0 and shape is (nlayer + 1), so after softmax it just becomes average weights
        #JO: The multiplication is (1, nlayer + 1) * (B, L, nlayer + 1, C) and result is (B, L, 1, C)
        #JO: Final esm_s is (B, L, C), achieving effect that all the layer is averaged
        #JO: Output of ADK1, [1, 214, 2560]
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        print("After combination, shape of the result of ESM2 calculation",esm_s.shape)

        return esm_s, aa, B, L, residx, mask, num_recycles
    
    def get_structure(self, esm_s, aa, B, L, residx, mask, num_recycles):
        with torch.no_grad():
            #JO: Norm -> Linear -> ReLU -> Linear, output shape is (B, L, CS)
            s_s_0 = self.esm_s_mlp(esm_s)
            print("Successfullt pass the mlp layer in Folding Trunk!!!")
            #JO: The s_z_0 shape is (B, L, L, CZ)
            s_z_0 = s_s_0.new_zeros(B, L, L, self.cfg.trunk.pairwise_state_dim)

            s_s_0 += self.embedding(aa)
            print("After adding the embedding, s_s_0: ",s_s_0)
            #JO: This is the last mask here in esmfold
            structure: dict = self.trunk(
                s_s_0, s_z_0, aa, residx, mask, no_recycles=num_recycles
            )
            # Documenting what we expect:
            structure = {
                k: v
                for k, v in structure.items()
                if k
                in [
                    "s_z",
                    "s_s",
                    "frames",
                    "sidechain_frames",
                    "unnormalized_angles",
                    "angles",
                    "positions",
                    "states",
                ]
            }

            disto_logits = self.distogram_head(structure["s_z"])
            disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
            structure["distogram_logits"] = disto_logits

            lm_logits = self.lm_head(structure["s_s"])
            structure["lm_logits"] = lm_logits

            structure["aatype"] = aa
            make_atom14_masks(structure)

            for k in [
                "atom14_atom_exists",
                "atom37_atom_exists",
            ]:
                structure[k] *= mask.unsqueeze(-1)
            structure["residue_index"] = residx

            lddt_head = self.lddt_head(structure["states"]).reshape(
                structure["states"].shape[0], B, L, -1, self.lddt_bins
            )
            structure["lddt_head"] = lddt_head
            plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
            structure["plddt"] = 100 * plddt  # we predict plDDT between 0 and 1, scale to be between 0 and 100.
            #JO: If I can get to know what each dimention of lddt_head is, I can understand the model better
            ptm_logits = self.ptm_head(structure["s_z"])

            seqlen = mask.type(torch.int64).sum(1)
            structure["ptm_logits"] = ptm_logits
            structure["ptm"] = torch.stack([
                compute_tm(batch_ptm_logits[None, :sl, :sl], max_bins=31, no_bins=self.distogram_bins)
                for batch_ptm_logits, sl in zip(ptm_logits, seqlen)
            ])
            structure.update(
                compute_predicted_aligned_error(
                    ptm_logits, max_bin=31, no_bins=self.distogram_bins
                )
            )

            return structure

    @torch.no_grad()
    def infer(
        #JO: Optional means the parameter can be None, Union means the parameter can be either of the types
        self,
        sequences: T.Union[str, T.List[str]],
        residx=None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = None,
        residue_index_offset: T.Optional[int] = 512,
        chain_linker: T.Optional[str] = "G" * 25,
    ):
        """Runs a forward pass given input sequences.

        Args:
            sequences (Union[str, List[str]]): A list of sequences to make predictions for. Multimers can also be passed in,
                each chain should be separated by a ':' token (e.g. "<chain1>:<chain2>:<chain3>").
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles (cfg.trunk.max_recycles), which is 4.
            residue_index_offset (int): Residue index separation between chains if predicting a multimer. Has no effect on
                single chain predictions. Default: 512.
            chain_linker (str): Linker to use between chains if predicting a multimer. Has no effect on single chain
                predictions. Default: length-25 poly-G ("G" * 25).
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        '''
        JOJO: aatype is the stacked encoded sequences (amino acid in each position in a list, but have been encoded
        according to openfold residue_constants.restype_order_with_x.get), 
        and mask is the initialized mask just the same shape as aatype (all is one)
        residx is the number of index of residues, linker_mask and chain_index does not make a big
        difference in the monomer case
        '''
        aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
            sequences, residue_index_offset, chain_linker
        )

        if residx is None:
            residx = _residx
        elif not isinstance(residx, torch.Tensor):
            residx = collate_dense_tensors(residx)

        aatype, mask, residx, linker_mask = map(
            lambda x: x.to(self.device), (aatype, mask, residx, linker_mask)
        )
        print("aatype before input into forward function:", aatype)
        print("B is its first dimention, L is its second dimention")
        print("mask before input into forward function, which should be all one:", mask)
        esm_s, aa, B, L, residx, mask, num_recycles = self.forward(
            aatype,
            mask=mask,
            residx=residx,
            masking_pattern=masking_pattern,
            num_recycles=num_recycles,
        )

        return esm_s, aa, B, L, residx, mask, num_recycles, linker_mask, chain_index
    
    def infer_structure(self, esm_s, aa, B, L, residx, mask, num_recycles, linker_mask, chain_index):
        output = self.get_structure(esm_s, aa, B, L, residx, mask, num_recycles)
        output["atom37_atom_exists"] = output[
            "atom37_atom_exists"
        ] * linker_mask.unsqueeze(2)

        output["mean_plddt"] = (output["plddt"] * output["atom37_atom_exists"]).sum(
            dim=(1, 2)
        ) / output["atom37_atom_exists"].sum(dim=(1, 2))
        output["chain_index"] = chain_index

        return output

    def output_to_pdb(self, output: T.Dict) -> T.List[str]:
        """Returns the pbd (file) string from the model given the model output."""
        return output_to_pdb(output)

    def infer_pdbs(self, seqs: T.List[str], *args, **kwargs) -> T.List[str]:
        """Returns list of pdb (files) strings from the model given a list of input sequences."""
        output = self.infer(seqs, *args, **kwargs)
        return self.output_to_pdb(output)

    def infer_pdb(self, sequence: str, *args, **kwargs) -> str:
        """Returns the pdb (file) string from the model given an input sequence."""
        return self.infer_pdbs([sequence], *args, **kwargs)[0]

    def set_chunk_size(self, chunk_size: T.Optional[int]):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        # Setting the value to None will return to default behavior, disable chunking.
        self.trunk.set_chunk_size(chunk_size)

    @property
    def device(self):
        return self.esm_s_combine.device
