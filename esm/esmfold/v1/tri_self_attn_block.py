# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from openfold.model.triangular_attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from torch import nn

from esm.esmfold.v1.misc import (
    Attention,
    Dropout,
    PairToSequence,
    ResidueMLP,
    SequenceToPair,
)


class TriangularSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        sequence_state_dim, #JO: 1024
        pairwise_state_dim, #JO: 128
        sequence_head_width, #JO: 32
        pairwise_head_width, #JO: 32
        dropout=0,
        **__kwargs,
    ):
        super().__init__()

        assert sequence_state_dim % sequence_head_width == 0
        assert pairwise_state_dim % pairwise_head_width == 0
        #JO: Reassign, Sequence_Num_heads: SH, Sequence_state_dim: SD, Sequence_head_width: SW
        sequence_num_heads = sequence_state_dim // sequence_head_width
        #JO: Reassign, Pairwise_Num_heads: PH, Pairwise_state_dim: PD, Pairwise_head_width: PW
        pairwise_num_heads = pairwise_state_dim // pairwise_head_width 
        assert sequence_state_dim == sequence_num_heads * sequence_head_width
        assert pairwise_state_dim == pairwise_num_heads * pairwise_head_width
        assert pairwise_state_dim % 2 == 0

        self.sequence_state_dim = sequence_state_dim
        self.pairwise_state_dim = pairwise_state_dim
        #JO: Normalization of last dimension of input (the same size as SD)
        self.layernorm_1 = nn.LayerNorm(sequence_state_dim)

        for param in self.layernorm_1.parameters():
            param.requires_grad = False

        self.sequence_to_pair = SequenceToPair(
            sequence_state_dim, pairwise_state_dim // 2, pairwise_state_dim
        )
        
        for param in self.sequence_to_pair.parameters():
            param.requires_grad = False
        
        self.pair_to_sequence = PairToSequence(pairwise_state_dim, sequence_num_heads)
        
        for param in self.pair_to_sequence.parameters():
            param.requires_grad = False
        
        self.seq_attention = Attention(
            sequence_state_dim, sequence_num_heads, sequence_head_width, gated=True
        )

        for param in self.seq_attention.parameters():
            param.requires_grad = False

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            pairwise_state_dim,
            pairwise_state_dim,
        )

        for param in self.tri_mul_out.parameters():
            param.requires_grad = False

        self.tri_mul_in = TriangleMultiplicationIncoming(
            pairwise_state_dim,
            pairwise_state_dim,
        )

        for param in self.tri_mul_in.parameters():
            param.requires_grad = False

        self.tri_att_start = TriangleAttentionStartingNode(
            pairwise_state_dim,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )  # type: ignore

        for param in self.tri_att_start.parameters():
            param.requires_grad = False

        self.tri_att_end = TriangleAttentionEndingNode(
            pairwise_state_dim,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )  # type: ignore

        for param in self.tri_att_end.parameters():
            param.requires_grad = False

        self.mlp_seq = ResidueMLP(sequence_state_dim, 4 * sequence_state_dim, dropout=dropout)

        for param in self.mlp_seq.parameters():
            param.requires_grad = False

        self.mlp_pair = ResidueMLP(pairwise_state_dim, 4 * pairwise_state_dim, dropout=dropout)

        for param in self.mlp_pair.parameters():
            param.requires_grad = False

        assert dropout < 0.4
        self.drop = nn.Dropout(dropout)
        self.row_drop = Dropout(dropout * 2, 2)
        self.col_drop = Dropout(dropout * 2, 1)

        torch.nn.init.zeros_(self.tri_mul_in.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_in.linear_z.bias)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.bias)
        torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.weight)
        torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.bias)
        torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.weight)
        torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.bias)

        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.weight)
        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.bias)
        torch.nn.init.zeros_(self.pair_to_sequence.linear.weight)
        torch.nn.init.zeros_(self.seq_attention.o_proj.weight)
        torch.nn.init.zeros_(self.seq_attention.o_proj.bias)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-2].weight)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-2].bias)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].weight)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].bias)

    def forward(self, sequence_state, pairwise_state, mask=None, chunk_size=None, **__kwargs):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim
          pairwise_state: B x L x L x pairwise_state_dim
          mask: B x L boolean tensor of valid positions

        Output:
          sequence_state: B x L x sequence_state_dim
          pairwise_state: B x L x L x pairwise_state_dim
        """
        assert len(sequence_state.shape) == 3
        assert len(pairwise_state.shape) == 4
        if mask is not None:
            assert len(mask.shape) == 2

        batch_dim, seq_dim, sequence_state_dim = sequence_state.shape
        pairwise_state_dim = pairwise_state.shape[3]
        assert sequence_state_dim == self.sequence_state_dim
        assert pairwise_state_dim == self.pairwise_state_dim
        assert batch_dim == pairwise_state.shape[0]
        assert seq_dim == pairwise_state.shape[1]
        assert seq_dim == pairwise_state.shape[2]

        # Update sequence state
        bias = self.pair_to_sequence(pairwise_state)

        # Self attention with bias + mlp.
        y = self.layernorm_1(sequence_state)
        y, _ = self.seq_attention(y, mask=mask, bias=bias)
        sequence_state = sequence_state + self.drop(y)
        sequence_state = self.mlp_seq(sequence_state)

        # Update pairwise state
        pairwise_state = pairwise_state + self.sequence_to_pair(sequence_state)

        # Axial attention with triangular bias.
        tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_mul_out(pairwise_state, mask=tri_mask)
        )
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_mul_in(pairwise_state, mask=tri_mask)
        )
        low_mem = dict(use_memory_efficient_kernel=True,
                       chunk_size=chunk_size)
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_att_start(pairwise_state, mask=tri_mask, **low_mem)
        )
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_att_end(pairwise_state, mask=tri_mask, **low_mem)
        )

        # MLP over pairs.
        # pairwise_state = self.mlp_pair(pairwise_state)

        # save memory
        self.mlp_pair.half()
        pairwise_state = self.mlp_pair(pairwise_state.half())
        pairwise_state = pairwise_state.to(sequence_state.dtype)

        return sequence_state, pairwise_state
