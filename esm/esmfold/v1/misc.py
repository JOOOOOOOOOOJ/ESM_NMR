# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import typing as T
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from openfold.np import residue_constants
from openfold.np.protein import Protein as OFProtein
from openfold.np.protein import to_pdb
from openfold.np import residue_constants
from openfold.utils.feats import atom14_to_atom37
# from esm.esmfold.v1.HydrogenBuilder import HydrogenBuilder
# from esm.esmfold.v1.legolas import EntryPDB, ChemicalShiftPredictor
# from esm.esmfold.v1.PdbBuilder import PdbBuilder


def encode_sequence(
    seq: str,
    residue_index_offset: T.Optional[int] = 512,
    chain_linker: T.Optional[str] = "G" * 25,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if chain_linker is None:
        chain_linker = ""
    if residue_index_offset is None:
        residue_index_offset = 0

    chains = seq.split(":")
    seq = chain_linker.join(chains)

    unk_idx = residue_constants.restype_order_with_x["X"]
    encoded = torch.tensor(
        [residue_constants.restype_order_with_x.get(aa, unk_idx) for aa in seq]
    )
    residx = torch.arange(len(encoded))

    if residue_index_offset > 0:
        start = 0
        for i, chain in enumerate(chains):
            residx[start : start + len(chain) + len(chain_linker)] += (
                i * residue_index_offset
            )
            start += len(chain) + len(chain_linker)

    linker_mask = torch.ones_like(encoded, dtype=torch.float32)
    chain_index = []
    offset = 0
    for i, chain in enumerate(chains):
        if i > 0:
            chain_index.extend([i - 1] * len(chain_linker))
        chain_index.extend([i] * len(chain))
        offset += len(chain)
        linker_mask[offset : offset + len(chain_linker)] = 0
        offset += len(chain_linker)

    chain_index = torch.tensor(chain_index, dtype=torch.int64)

    return encoded, residx, linker_mask, chain_index


def batch_encode_sequences(
    sequences: T.Sequence[str],
    residue_index_offset: T.Optional[int] = 512,
    chain_linker: T.Optional[str] = "G" * 25,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    aatype_list = []
    residx_list = []
    linker_mask_list = []
    chain_index_list = []
    print("This is sequence just after input: ", sequences)
    for seq in sequences:
        #JO: aatype_seq is the sequence put into list, residx_seq is the residue index, 
        #JO: linker_mask_seq and chain_index_seq are almost nothing in monomer case
        aatype_seq, residx_seq, linker_mask_seq, chain_index_seq = encode_sequence(
            seq,
            residue_index_offset=residue_index_offset,
            chain_linker=chain_linker,
        )
        aatype_list.append(aatype_seq)
        residx_list.append(residx_seq)
        linker_mask_list.append(linker_mask_seq)
        chain_index_list.append(chain_index_seq)
    print("This is the sequence after encoding: ",aatype_list)
    aatype = collate_dense_tensors(aatype_list)
    print("This is the sequence after stacking: ",aatype)
    mask = collate_dense_tensors(
        [aatype.new_ones(len(aatype_seq)) for aatype_seq in aatype_list]
    )
    print("Get prepared for the mask (initialization): ",mask)
    residx = collate_dense_tensors(residx_list)
    linker_mask = collate_dense_tensors(linker_mask_list)
    chain_index_list = collate_dense_tensors(chain_index_list, -1)

    return aatype, mask, residx, linker_mask, chain_index_list

def output_to_pdbh_download(seq, coords, path):
    pdb_creator = PdbBuilder(seq,
        coords.cpu().detach().numpy(),
        terminal_atoms=None,
        has_hydrogens=True)
    pdb_creator.save_pdb(path) #Can add title


def output_to_pdb_download(output: T.Dict) -> T.List[str]:
    """Returns the pbd (file) string from the model given the model output."""
    # atom14_to_atom37 must be called first, as it fails on latest numpy if the
    # input is a numpy array. It will work if the input is a torch tensor.
    final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
    output = {k: v.to("cpu").numpy() for k, v in output.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = output["atom37_atom_exists"]
    pdbs = []
    for i in range(output["aatype"].shape[0]):
        aa = output["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = output["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=output["plddt"][i],
            chain_index=output["chain_index"][i] if "chain_index" in output else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def output_to_pdb(output: T.Dict) -> T.List[str]:
    """Returns the pbd (file) string from the model given the model output."""
    # atom14_to_atom37 must be called first, as it fails on latest numpy if the
    # input is a numpy array. It will work if the input is a torch tensor.
    # final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
    #JO: atom14 is the style PDB uses, 37 is the style AlphaFold uses
    device_use = output["aligned_confidence_probs"].device
    final_atom_positions = output["positions"][-1]
    final_atom_positions = final_atom_positions.squeeze(0)
    print("Does final_atom_positions require gradient?",final_atom_positions.requires_grad)
    L = final_atom_positions.shape[0]
    new_column = torch.full((L, 1, 3), float('nan'), device=final_atom_positions.device, requires_grad=True) 
    atom_position_before_hydro = torch.cat([
        final_atom_positions[:, :4, :],
        new_column,
        final_atom_positions[:, 4:, :]
    ], dim=1)
    print("Does atom_position_before_hydro require gradient?",atom_position_before_hydro.requires_grad)
    #JO: Put all the items to numpy, cancel.
    # output = {k: v.to("cpu").numpy() for k, v in output.items()}
    # final_atom_positions = final_atom_positions.cpu().numpy()
    # final_atom_mask = output["atom37_atom_exists"]
    #JO: Use Openfold functions
    restypes = residue_constants.restypes + ["X"]
    aatype = output["aatype"].squeeze(0)  # 适用于 torch.Tensor 和 numpy.ndarray
    aa_seq = "".join([restypes[idx] for idx in aatype])
    coord_h = HydrogenBuilder(aa_seq, atom_position_before_hydro, device=device_use)
    atom_position_after_hydrogen = coord_h.build_hydrogens()
    print("Does atom_position_after_hydro require gradient?",atom_position_after_hydrogen.requires_grad)
    #Legolas
    NUM_MODELS = 5

    interested_atypes = ["N"]
    # Get the directory where models reside
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Define paths relative to the script's directory
    model_paths = {}
    for atype in interested_atypes:
        model_paths[atype] = [
            os.path.join(script_dir, "ens_models", f"ens_model_{i+1}_{atype}.pt") for i in range(NUM_MODELS)
        ]

    entry = EntryPDB(aa_seq, atom_position_after_hydrogen, device_use, interested_atypes)
    model = ChemicalShiftPredictor(model_paths, interested_atypes).to(device_use)
    cs_prediction_type, cs_prediction = model(entry, 10)
    for key, value in cs_prediction.items():
        if isinstance(value, torch.Tensor):
            print(f"Key: {key}, requires_grad: {value.requires_grad}")

    return aa_seq, atom_position_before_hydro, atom_position_after_hydrogen, cs_prediction_type, cs_prediction

#JO: try to stack the tensors in the sample list into a single tensor, also make sure they are the same shape
def collate_dense_tensors(
    samples: T.List[torch.Tensor], pad_v: float = 0
) -> torch.Tensor:
    """
    Takes a list of tensors with the following dimensions:
        [(d_11,       ...,           d_1K),
         (d_21,       ...,           d_2K),
         ...,
         (d_N1,       ...,           d_NK)]
    and stack + pads them into a single tensor of:
    (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
    """
    if len(samples) == 0:
        return torch.Tensor()
    if len(set(x.dim() for x in samples)) != 1:
        raise RuntimeError(
            f"Samples has varying dimensions: {[x.dim() for x in samples]}"
        )
    (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    result = torch.empty(
        len(samples), *max_shape, dtype=samples[0].dtype, device=device
    )
    result.fill_(pad_v)
    for i in range(len(samples)):
        result_i = result[i]
        t = samples[i]
        result_i[tuple(slice(0, k) for k in t.shape)] = t
    return result

#JO: input initiation: (SD, SH, SW)
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_width, gated=False):
        super().__init__()
        assert embed_dim == num_heads * head_width

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_width = head_width

        self.proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gated = gated #JO: True
        if gated:
            self.g_proj = nn.Linear(embed_dim, embed_dim)
            torch.nn.init.zeros_(self.g_proj.weight)
            torch.nn.init.ones_(self.g_proj.bias)

        self.rescale_factor = self.head_width**-0.5

        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, x, mask=None, bias=None, indices=None):
        """
        Basic self attention with optional mask and external pairwise bias.
        To handle sequences of different lengths, use mask.

        Inputs:
          x: batch of input sequneces (.. x L x C) C: embed_dim
          mask: batch of boolean masks where 1=valid, 0=padding position (.. x L_k). optional.
          bias: batch of scalar pairwise attention biases (.. x Lq x Lk x num_heads). optional.

        Outputs:
          sequence projection (B x L x embed_dim), attention maps (B x L x L x num_heads)
        """
        #JO: h is num heads, l is sequence length, c is head width
        t = rearrange(self.proj(x), "... l (h c) -> ... h l c", h=self.num_heads)
        q, k, v = t.chunk(3, dim=-1)

        q = self.rescale_factor * q
        a = torch.einsum("...qc,...kc->...qk", q, k)

        # Add external attention bias.
        if bias is not None:
            a = a + rearrange(bias, "... lq lk h -> ... h lq lk")

        # Do not attend to padding tokens.
        if mask is not None:
            mask = repeat(
                mask, "... lk -> ... h lq lk", h=self.num_heads, lq=q.shape[-2]
            )
            a = a.masked_fill(mask == False, -np.inf)

        a = F.softmax(a, dim=-1)

        y = torch.einsum("...hqk,...hkc->...qhc", a, v)
        y = rearrange(y, "... h c -> ... (h c)", h=self.num_heads)

        if self.gated:
            y = self.g_proj(x).sigmoid() * y
        y = self.o_proj(y)

        return y, rearrange(a, "... lq lk h -> ... h lq lk")


class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.
    """

    def __init__(self, r: float, batch_dim: T.Union[int, T.List[int]]):
        super(Dropout, self).__init__()

        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        return x * self.dropout(x.new_ones(shape))

#JO: input initiation: (SD, inner_dim = PD//2, PD)
class SequenceToPair(nn.Module):
    def __init__(self, sequence_state_dim, inner_dim, pairwise_state_dim):
        super().__init__()

        self.layernorm = nn.LayerNorm(sequence_state_dim)
        #JO: So these two steps change shape from SD to PD then to PD
        self.proj = nn.Linear(sequence_state_dim, inner_dim * 2, bias=True)
        self.o_proj = nn.Linear(2 * inner_dim, pairwise_state_dim, bias=True)
        #JO: initialize the bias to zero
        torch.nn.init.zeros_(self.proj.bias)
        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, sequence_state):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim

        Output:
          pairwise_state: B x L x L x pairwise_state_dim

        Intermediate state:
          B x L x L x 2*inner_dim
        """

        assert len(sequence_state.shape) == 3

        s = self.layernorm(sequence_state)
        s = self.proj(s)
        q, k = s.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]

        x = torch.cat([prod, diff], dim=-1)
        x = self.o_proj(x)

        return x

#JO: input initiation: (PD, SH)
class PairToSequence(nn.Module):
    def __init__(self, pairwise_state_dim, num_heads):
        super().__init__()
        #JO: Change shape from PD to SH
        self.layernorm = nn.LayerNorm(pairwise_state_dim)
        self.linear = nn.Linear(pairwise_state_dim, num_heads, bias=False)

    def forward(self, pairwise_state):
        """
        Inputs:
          pairwise_state: B x L x L x pairwise_state_dim

        Output:
          pairwise_bias: B x L x L x num_heads
        """
        assert len(pairwise_state.shape) == 4
        z = self.layernorm(pairwise_state)
        pairwise_bias = self.linear(z)
        return pairwise_bias


class ResidueMLP(nn.Module):
    def __init__(self, embed_dim, inner_dim, norm=nn.LayerNorm, dropout=0):
        super().__init__()

        self.mlp = nn.Sequential(
            norm(embed_dim),
            nn.Linear(embed_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.mlp(x)
