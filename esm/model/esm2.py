# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union
import torch
import torch.nn as nn

import esm
from esm.modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer


class ESM2(nn.Module):
    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,
    ):
        print("Number of layers used in ESM2 initialization: ", num_layers)
        print("It's also the number of Transformer layers")
        print("Embedding dimension used in ESM2 initialization: ", embed_dim)
        print("Number of attention heads used in ESM2 initialization: ", attention_heads)
        print("Alphabet used in ESM2 initialization: ", alphabet)
        print("Token dropout used in ESM2 initialization: ", token_dropout)
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        #JOJO: Some prepend tokens
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        #JOJO: beginning  and end of sequence tokens
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        print("Padding index, mask index, cls index, eos index used in ESM2 alphabet: ", self.padding_idx, self.mask_idx, self.cls_idx, self.eos_idx)
        ''' 
        JOJO's Question:
        what kind of influence will the token_dropout have on the model?
        '''
        self.token_dropout = token_dropout
        # self.token_dropout = False

        self._init_submodules()

    def _init_submodules(self):
        self.embed_scale = 1
        #JO: It's a dictionary here, and the weight from pre-trained model has been loaded
        #JO: This is the embedding first to make the input sequence into a sequence of vectors
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )
        #JO: embed dim and attention heads come from config
        #JO: Number of Transformer Layer is num_layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        #The last three parameters are from alphabet
        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )
    #JO: tokens are the indexes list with bos, padding, eos added
    #JO: repr_layers is the list from 0 to num_layers including the end (num_layers + 1)
    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        #JO: I think sometime I will need the return contacts
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2 # B, L
        #JO: generate the padding mask the same size as tokens with position of padding marked as True [1, L]
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        #JO: First step, apply embedding to the tokens, and then apply the scale.
        #JO: Now the shape of x is [B, L, C], C is the embed_dimention
        x = self.embed_scale * self.embed_tokens(tokens)
        #JO: True in config, but this dropout is not disgarding the value
        if self.token_dropout:
            '''
            JOJO: Here the position in token which has mask idx is broadcasted to embedding dimention
            and set all that dimention to 0
            '''
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            #JO: The true length of the sequence without padding
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            #JO: Do scaling to x, but I don't very understand the logic here
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        #JO: Make the padding mask position as 0 (The C dimention is all 0)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        #JO: This is the layer that to be extracted
        repr_layers = set(repr_layers)
        hidden_representations = {}
        #JO: There are num_layer + 1 layers 
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        #JO: (B, L, C) => (L, B, C)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None
        #JO: In this loop, the x is updated by the TransformerLayer. There are num_layers layers in total
        #JO: Here the padding mask takes effect again.
        for layer_idx, layer in enumerate(self.layers):
            #JO: x shape is (L, B, C) and attn (weights) shape is (N, B, L, L)
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            #JO: So the hidden representations are the results of each layer, shape is (B, L, C)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E) #JO: The shape of x is (B, L, C)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)
        #JO: The Shape of x is (B, L, A), A is the alphabet size
        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]
