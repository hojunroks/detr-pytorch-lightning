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
        self.self_attention = nn.MultiheadAttention(embed_dim=,
                                                    num_heads=,
                                                    dropout=
                                                    )
        self.norm1 = nn.LayerNorm(normalized_shape=)
        self.ffn = nn.Sequential(


        )
        self.norm2 = nn.LayerNorm(normalized_shape=)


    def forward(self, features, pos_encoding):
        V =

class TransformerDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):


