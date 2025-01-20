# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as T
from contextlib import ExitStack
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from openfold.model.structure_module import StructureModule

from esm.esmfold.v1.tri_self_attn_block import TriangularSelfAttentionBlock


@dataclass
class StructureModuleConfig:
    c_s: int = 384
    c_z: int = 128
    c_ipa: int = 16
    c_resnet: int = 128
    no_heads_ipa: int = 12
    no_qk_points: int = 4
    no_v_points: int = 8
    dropout_rate: float = 0.0
    no_blocks: int = 8
    no_transition_layers: int = 1
    no_resnet_blocks: int = 2
    no_angles: int = 7
    trans_scale_factor: int = 10
    epsilon: float = 1e-8
    inf: float = 1e5

"""
JOJO: Folding Trunk Config has higher superiority than Structure Module Config
"""

@dataclass
class FoldingTrunkConfig:
    _name: str = "FoldingTrunkConfig"
    num_blocks: int = 48
    sequence_state_dim: int = 1024
    pairwise_state_dim: int = 128
    sequence_head_width: int = 32
    pairwise_head_width: int = 32
    position_bins: int = 32
    dropout: float = 0.0
    layer_drop: float = 0
    cpu_grad_checkpoint: bool = False

    max_recycles: int = 4
    chunk_size: T.Optional[int] = None

    structure_module: StructureModuleConfig = field(default_factory=StructureModuleConfig) #JO: Debug ESMFold, this is the characteristics of dataclass


def get_axial_mask(mask):
    """
    Helper to convert B x L mask of valid positions to axial mask used
    in row column attentions.

    Input:
      mask: B x L tensor of booleans

    Output:
      mask: B x L x L tensor of booleans
    """

    if mask is None:
        return None
    assert len(mask.shape) == 2
    batch_dim, seq_dim = mask.shape
    m = mask.unsqueeze(1).expand(batch_dim, seq_dim, seq_dim)
    m = m.reshape(batch_dim * seq_dim, seq_dim)
    return m

#JO: Output Position Embedding (Normalization)
class RelativePosition(nn.Module):
    def __init__(self, bins, pairwise_state_dim):
        super().__init__()
        self.bins = bins

        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        #JO: Embedding is used to convert the residue index to pairwise state dimension: 2BIN+2 -> CZ
        self.embedding = torch.nn.Embedding(2 * bins + 2, pairwise_state_dim)
        for param in self.embedding.parameters():
            param.requires_grad = False 

    def forward(self, residue_index, mask=None):
        """
        Input:
          residue_index: B x L tensor of indices (dytpe=torch.long)
          mask: B x L tensor of booleans
        JO: B is batch size, L is sequence length
        Output:
          pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """

        assert residue_index.dtype == torch.long
        if mask is not None:
            assert residue_index.shape == mask.shape
        #JO: Normalization comes from here
        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # Add 1 to adjust for padding index.

        if mask is not None:
            mask = mask[:, None, :] * mask[:, :, None]
            diff[mask == False] = 0

        output = self.embedding(diff)
        return output


class FoldingTrunk(nn.Module):
    #JO: Is called by "self.trunk = FoldingTrunk(**cfg.trunk)"
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = FoldingTrunkConfig(**kwargs)
        print("folding trunk dropout: ", self.cfg.dropout)
        assert self.cfg.max_recycles > 0
        #JO: Here CS and CZ are from config file, hope it will be the same as trunk cs and cz
        c_s = self.cfg.sequence_state_dim
        c_z = self.cfg.pairwise_state_dim
        print("cz (sequence state dim) and cs (pairwise state dim) inside the folding trunk, hope it will be the same as outside: ", c_s, c_z)
        print("sequence head width and pairwise head width: ", self.cfg.sequence_head_width, self.cfg.pairwise_head_width)
        #JO: MultiHead again
        assert c_s % self.cfg.sequence_head_width == 0
        assert c_z % self.cfg.pairwise_head_width == 0
        block = TriangularSelfAttentionBlock

        print("position bins used in position embeddings: ", self.cfg.position_bins) #JO: 32
        self.pairwise_positional_embedding = RelativePosition(self.cfg.position_bins, c_z)
        #JO: Look like the MultiHeadAttention is replaced by TriangularSelfAttentionBlock
        print("number of blocks in folding trunk (TriangularSelfAttentionBlock): ", self.cfg.num_blocks) #JO: 48
        self.blocks = nn.ModuleList(
            [
                block(
                    sequence_state_dim=c_s,
                    pairwise_state_dim=c_z,
                    sequence_head_width=self.cfg.sequence_head_width,
                    pairwise_head_width=self.cfg.pairwise_head_width,
                    dropout=self.cfg.dropout,
                    # dropout=0.15, #JOJO's changes
                )
                for i in range(self.cfg.num_blocks)
            ]
        )

        # Freeze parameters of TriangularSelfAttentionBlock modules
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.recycle_bins = 15
        self.recycle_s_norm = nn.LayerNorm(c_s)
        for param in self.recycle_s_norm.parameters():
            param.requires_grad = False
        self.recycle_z_norm = nn.LayerNorm(c_z)
        for param in self.recycle_z_norm.parameters():
            param.requires_grad = False
        self.recycle_disto = nn.Embedding(self.recycle_bins, c_z)
        self.recycle_disto.weight[0].detach().zero_()
        for param in self.recycle_disto.parameters():
            param.requires_grad = False
        #JO: Test
        # self.cfg.structure_module.dropout_rate = 0.2
        self.structure_module = StructureModule(**self.cfg.structure_module)  # type: ignore
        #JO: Freeze the parameters of the structure module
        for param in self.structure_module.parameters():
            param.requires_grad = False
        # print(self.cfg.structure_module)
        self.trunk2sm_s = nn.Linear(c_s, self.structure_module.c_s)
        self.trunk2sm_z = nn.Linear(c_z, self.structure_module.c_z)
        # freeze trunk2sm_s and trunk2sm_z
        for param in self.trunk2sm_s.parameters():
            param.requires_grad = False
        for param in self.trunk2sm_z.parameters():
            param.requires_grad = False
        self.chunk_size = self.cfg.chunk_size

    def set_chunk_size(self, chunk_size):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        self.chunk_size = chunk_size

    def forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles: T.Optional[int] = None):
        """
        Inputs:
          seq_feats:     B x L x C            tensor of sequence features
          pair_feats:    B x L x L x C        tensor of pair features
          residx:        B x L                long tensor giving the position in the sequence
          mask:          B x L                boolean tensor indicating valid residues

        Output:
          predicted_structure: B x L x (num_atoms_per_residue * 3) tensor wrapped in a Coordinates object
        """

        device = seq_feats.device
        s_s_0 = seq_feats
        s_z_0 = pair_feats

        if no_recycles is None:
            no_recycles = self.cfg.max_recycles
        else:
            assert no_recycles >= 0, "Number of recycles must not be negative."
            no_recycles += 1  # First 'recycle' is just the standard forward pass through the model.

        def trunk_iter(s, z, residx, mask):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)

            for block in self.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            return s, z
        print("Successfully arrive before the recycle loop in Folding Trunk")
        #############JO: Check memory usage#############
        device = torch.device("cuda:0")  # 假设使用第一块GPU

        # 获取GPU总内存
        total_memory = torch.cuda.get_device_properties(device).total_memory

        # 获取当前已分配的内存
        allocated_memory = torch.cuda.memory_allocated(device)

        # 计算剩余内存
        free_memory = total_memory - allocated_memory

        print(f'GPU total memory: {total_memory / 1024 ** 2:.2f} MB')
        print(f'Allocated memory: {allocated_memory / 1024 ** 2:.2f} MB')
        print(f'Remaining memory: {free_memory / 1024 ** 2:.2f} MB')
        #############JO: Check memory usage#############
        s_s = s_s_0
        s_z = s_z_0
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

        assert no_recycles > 0
        for recycle_idx in range(no_recycles):
            with ExitStack() if recycle_idx == no_recycles - 1 else torch.no_grad():
                # === Recycling ===
                recycle_s = self.recycle_s_norm(recycle_s.detach())
                recycle_z = self.recycle_z_norm(recycle_z.detach())
                recycle_z += self.recycle_disto(recycle_bins.detach())
                print("Successfully arrive before the trunk_iter in Folding Trunk")
                #############JO: Check memory usage#############
                device = torch.device("cuda:0")  # 假设使用第一块GPU

                # 获取GPU总内存
                total_memory = torch.cuda.get_device_properties(device).total_memory

                # 获取当前已分配的内存
                allocated_memory = torch.cuda.memory_allocated(device)

                # 计算剩余内存
                free_memory = total_memory - allocated_memory

                print(f'GPU total memory: {total_memory / 1024 ** 2:.2f} MB')
                print(f'Allocated memory: {allocated_memory / 1024 ** 2:.2f} MB')
                print(f'Remaining memory: {free_memory / 1024 ** 2:.2f} MB')
                #############JO: Check memory usage#############
                with torch.no_grad():
                    s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)

                print("Successfully arrive after the trunk_iter in Folding Trunk")
                #############JO: Check memory usage#############
                device = torch.device("cuda:0")  # 假设使用第一块GPU

                # 获取GPU总内存
                total_memory = torch.cuda.get_device_properties(device).total_memory

                # 获取当前已分配的内存
                allocated_memory = torch.cuda.memory_allocated(device)

                # 计算剩余内存
                free_memory = total_memory - allocated_memory

                print(f'GPU total memory: {total_memory / 1024 ** 2:.2f} MB')
                print(f'Allocated memory: {allocated_memory / 1024 ** 2:.2f} MB')
                print(f'Remaining memory: {free_memory / 1024 ** 2:.2f} MB')
                #############JO: Check memory usage#############
                # === Structure module ===
                structure = self.structure_module(
                    {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                    true_aa,
                    mask.float(),
                )

                recycle_s = s_s
                recycle_z = s_z
                # Distogram needs the N, CA, C coordinates, and bin constants same as alphafold.
                recycle_bins = FoldingTrunk.distogram(
                    structure["positions"][-1][:, :, :3],
                    3.375,
                    21.375,
                    self.recycle_bins,
                )

        assert isinstance(structure, dict)  # type: ignore
        structure["s_s"] = s_s
        structure["s_z"] = s_z

        return structure

    @staticmethod
    def distogram(coords, min_bin, max_bin, num_bins):
        # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
        boundaries = torch.linspace(
            min_bin,
            max_bin,
            num_bins - 1,
            device=coords.device,
        )
        boundaries = boundaries**2
        N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]
        # Infer CB coordinates.
        b = CA - N
        c = C - CA
        a = b.cross(c, dim=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        dists = (CB[..., None, :, :] - CB[..., :, None, :]).pow(2).sum(dim=-1, keepdims=True)
        bins = torch.sum(dists > boundaries, dim=-1)  # [..., L, L]
        return bins
