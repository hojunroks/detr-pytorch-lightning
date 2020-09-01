import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):


class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):



class TransformerEncoderLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()

        # Multihead Self-attention Layer
        self.self_attention = nn.MultiheadAttention(embed_dim=,
                                                    num_heads=,
                                                    dropout=
                                                    )

        # First Add & Normalize Layer
        self.dropout1 = nn.Dropout()
        self.norm1 = nn.LayerNorm(normalized_shape=)

        # FFN Layer
        self.ffn = nn.Sequential(
            nn.Linear(),
            self.activation,
            nn.Dropout(),
            nn.Linear()
        )

        # Second Add & Normalize Layer
        self.dropout2 = nn.Dropout()
        self.norm2 = nn.LayerNorm(normalized_shape=)

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

    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()

        # Multihead Self-attention Layer
        self.self_attention = nn.MultiheadAttention(embed_dim=,
                                                    num_heads=,
                                                    dropout=
                                                    )
        # First Add & Normalize Layer
        self.dropout1 = nn.Dropout()
        self.norm1 = nn.LayerNorm(normalized_shape=)

        # Multihead attention Layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=,
                                                    num_heads=,
                                                    dropout=
                                                    )

        # Second Add & Normalize Layer
        self.dropout2 = nn.Dropout()
        self.norm2 = nn.LayerNorm(normalized_shape=)

        # FFN Layer
        self.ffn = nn.Sequential(
            nn.Linear(),
            self.activation,
            nn.Dropout(),
            nn.Linear()
        )

        # Third Add & Normalize Layer
        self.dropout3 = nn.Dropout()
        self.norm3 = nn.LayerNorm(normalized_shape=)


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
