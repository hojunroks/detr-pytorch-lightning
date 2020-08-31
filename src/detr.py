import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from .transformer import TransformerEncoder, TransformerDecoder

class DETR(LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.pos_encoding = PosEncoding()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
        self.obj_queries = ObjectQueries()
        self.pred_heads = PredictionHeads()

    def forward(self, src):

        pass

    def configure_optimizers(self):
        pass

    def training_step(self):
        pass