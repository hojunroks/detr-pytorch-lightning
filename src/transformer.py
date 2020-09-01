"""
DETR Transformer class.
Copy-paste from DETR.Transformer with modifications:
    * Change some module names and order regarding the paper
    * Remove some parameters and options for simplification
"""
from typing import Any

import torch
import torch.nn as nn
import copy


# Function that returns copies of modules
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Transformer(nn.Module):
    def __init__(self, n_dim, n_head, num_encoder_layers,
                 num_decoder_layers, ffn_hid, dropout
                 ):
        super().__init__()
        # Encoder
        encoder_layer = TransformerEncoderLayer(n_dim, n_head, ffn_hid, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        # Decoder
        decoder_layer = TransformerDecoderLayer(n_dim, n_head, ffn_hid, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, query_pos, pos_enc):
        '''
        Size of params
        features: BSxDxHxW where BS is batch size and D is embedding dimension size(=channel size)
        query_pos: NxD where N is the number of queries(=number of boxes to predict)
        pos_enc: BSxDxHxW
        :return: NxBSxD
        '''


        # flatten BSxDxHxW to HWxBSxD
        bs, d, h, w = features.shape
        features = features.flatten(2).permute(2, 0, 1)
        pos_enc = pos_enc.flatten(2).permute(2, 0, 1)

        query_pos = query_pos.unsqueeze(1).repeat(1, bs, 1)  # Size is now NxBSxD (Clone queries by batch size)
        queries = torch.zeros_like(query_pos)  # NxBSxD,initialized with zero
        enc_out = self.encoder(features, pos_enc)  # HWxBSxD
        hs = self.decoder(queries, query_pos, pos_enc, enc_out)  # NxBSxD without unnecessary unsqueeze
        # return hs.transpose(1, 2), enc_out.permute(1, 2, 0).view(bs, d, h, w)  # 1xBSxNxD, BSxDxHxW
        return hs.transpose(0, 1), enc_out.permute(1, 2, 0).view(bs, d, h, w)  # BSxNxD, BSxDxHxW


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, features, pos_enc):
        output = features
        for layer in self.layers:
            output = layer(output, pos_enc)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, queries, query_pos, pos_enc, enc_out):
        output = queries
        for layer in self.layers:
            output = layer(output, query_pos, pos_enc, enc_out)
        #return output.unsqueeze(0) Why did facebook unsqueeze this?
        return output



class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_dim, n_heads, ffn_hid, dropout):
        super().__init__()
        self.activation = nn.ReLU()

        # Multihead Self-attention Layer
        self.self_attention = nn.MultiheadAttention(embed_dim=n_dim,
                                                    num_heads=n_heads,
                                                    dropout=dropout
                                                    )

        # First Add & Normalize Layer
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(normalized_shape=n_dim)

        # FFN Layer
        self.ffn = nn.Sequential(
            nn.Linear(n_dim, ffn_hid),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(ffn_hid, n_dim)
        )

        # Second Add & Normalize Layer
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(normalized_shape=n_dim)

    def forward(self, features, pos_enc):
        V = features
        K = features + pos_enc
        Q = features + pos_enc

        # Multihead self-attention Layer
        x1 = self.self_attention(Q, K, V)[0]
        # First Add & Normalize Layer
        x1 = self.norm1(features+self.dropout1(x1))
        # FFN Layer
        x2 = self.ffn(x1)
        # Second Add & Normalize Layer
        x2 = self.norm2(x1+self.dropout2(x2))
        return x2


class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_dim, n_head, ffn_hid, dropout):
        super().__init__()
        self.activation = nn.ReLU()

        # Multihead Self-attention Layer
        self.self_attention = nn.MultiheadAttention(embed_dim=n_dim,
                                                    num_heads=n_head,
                                                    dropout=dropout
                                                    )
        # First Add & Normalize Layer
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(normalized_shape=n_dim)

        # Multihead attention Layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=n_dim,
                                                    num_heads=n_head,
                                                    dropout=dropout
                                                    )

        # Second Add & Normalize Layer
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(normalized_shape=n_dim)

        # FFN Layer
        self.ffn = nn.Sequential(
            nn.Linear(n_dim, ffn_hid),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(ffn_hid, n_dim)
        )

        # Third Add & Normalize Layer
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(normalized_shape=n_dim)

    def forward(self, queries, query_pos, pos_enc, enc_out):
        V1 = queries
        K1 = queries + query_pos
        Q1 = queries + query_pos

        # Multihead Self-attention Layer
        x1 = self.self_attention(Q1, K1, V1)[0]
        # First Add & Normalize Layer
        x1 = self.norm1(queries+self.dropout1(x1))

        V2 = enc_out
        K2 = enc_out + pos_enc
        Q2 = x1 + query_pos

        # Multihead attention Layer
        x2 = self.multihead_attention(Q2, K2, V2)[0]
        # Second Add & Normalize Layer
        x2 = self.norm2(x1 + self.dropout2(x2))
        # FFN Layer
        x3 = self.ffn(x2)
        # Second Add & Normalize Layer
        x3 = self.norm3(x2 + self.dropout3(x3))
