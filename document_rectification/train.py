import logging
import os
from argparse import Namespace

import pytorch_lightning as pl
import torch
import torchvision
from ez_torch.models import SpatialUVOffsetTransformer
from torch.nn import functional as F

from document_rectification import data

logger = logging.getLogger()


class GeometricTransformModel(pl.LightningModule):
    def __init__(self, res_w, res_h):
        super().__init__()
        self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        self.st = SpatialUVOffsetTransformer(
            i=1000,
            uv_resolution_shape=(res_w, res_h),
        )

    def forward(self, x):
        self.features = self.feature_extractor(x)
        x = x.mean(dim=1, keepdim=True)
        y_hat = self.st([self.features, x])
        return y_hat

    def criterion(self, y_hat, y):
        y = y.mean(dim=1, keepdim=True)
        assert y.shape == y_hat.shape
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, _batch_index):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss


def main():
    DEVICE = "cuda"
    model = GeometricTransformModel(res_w=20, res_h=20)
    model = model.to(DEVICE)
    datamodule = data.get_datamodule(train_bs=16, val_bs=16, plot_bs=8)

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
