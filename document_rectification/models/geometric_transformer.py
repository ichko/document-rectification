import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import wandb
from document_rectification.common import DEVICE
from document_rectification.data import DocumentsDataModule
from ez_torch.models import SpatialUVOffsetTransformer
from ez_torch.vis import Fig
from torchvision.models.mobilenetv2 import MobileNetV2


class GeometricTransformModel(pl.LightningModule):
    def __init__(self, res_w, res_h):
        super().__init__()
        self.feature_extractor = MobileNetV2(
            pretrained=True,
            num_classes=1000,
        )
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
        return F.binary_cross_entropy(y_hat, y)


def sanity_check():
    datamodule = DocumentsDataModule(
        train_bs=8,
        val_bs=8,
        plot_bs=8,
        shuffle=False,
        device=DEVICE,
    )
    model = GeometricTransformModel(
        res_w=5,
        res_h=5,
        datamodule=datamodule,
    ).to(DEVICE)

    dl = datamodule.plot_dataloader()
    batch = next(iter(dl))

    x = batch["x"]
    y = model(x)

    print(x.shape, y.shape)

    fig = Fig(nr=1, nc=3, figsize=(15, 10))
    im = batch["x"].ez.grid(nr=2).channel_last.np
    fig[0].imshow(im)
    fig[0].ax.set_title("Input")

    im = batch["y"].ez.grid(nr=2).channel_last.np
    fig[1].imshow(im)
    fig[1].ax.set_title("Output")

    im = y.ez.grid(nr=2).channel_last.np
    fig[2].imshow(im)
    fig[2].ax.set_title("Predictions")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sanity_check()
