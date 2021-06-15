from typing import Any

import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
from document_rectification.common import DEVICE
from document_rectification.data import get_datamodule
from ez_torch.models import Reshape
from pytorch_lightning.core import datamodule
from torch import nn
from torch.functional import Tensor


class Encoder(pl.LightningModule):
    def __init__(self, image_size):
        super().__init__()
        H, W = image_size
        self.net = torchvision.models.mobilenet.mobilenet_v2(
            pretrained=False,
            progress=True,
            num_classes=H * W,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(pl.LightningModule):
    def __init__(self, image_channels, image_size):
        super().__init__()
        H, W = image_size
        self.net = nn.Sequential(
            Reshape(-1, 1, H, W),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=1.25),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, image_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AutoEncoder(pl.LightningModule):
    def __init__(self, image_channels, image_size):
        super().__init__()
        self.encoder = Encoder(image_size=image_size)
        self.decoder = Decoder(
            image_channels=image_channels,
            image_size=image_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def criterion(self, y_hat, y):
        y = y.mean(dim=1, keepdim=True)
        return F.binary_cross_entropy(y_hat, y)


def sanity_check():
    dm = get_datamodule(
        train_bs=12,
        val_bs=32,
        plot_bs=16,
        shuffle=False,
        device=DEVICE,
    )
    size = [dm.H // 10, dm.W // 10]
    dl = dm.plot_dataloader()
    batch = next(iter(dl))

    ae = AutoEncoder(
        image_channels=3,
        image_size=size,
    ).to(DEVICE)
    ae.summarize()

    y_hat = ae.encoder(batch["x"])
    x_hat = ae.decoder(y_hat)

    print(x_hat.shape)
    print(y_hat.shape)


if __name__ == "__main__":
    sanity_check()
