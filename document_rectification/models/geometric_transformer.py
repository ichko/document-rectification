import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from document_rectification.common import DEVICE
from document_rectification.data import DocumentsDataModule
from ez_torch.models import SpatialLinearTransformer, SpatialUVOffsetTransformer
from ez_torch.vis import Fig
from torch import nn
from torch.functional import Tensor
from torchvision.models.mobilenetv2 import mobilenet_v2


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            # nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            # nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            # nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 5, stride=2, padding=1),
            # nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 5, stride=1, padding=0),
            # nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=0),
            # nn.BatchNorm2d(128, 0.8),
            # nn.AdaptiveAvgPool2d((32, 16)),
            nn.Conv2d(128, 1, 1, stride=1, padding=0),
            nn.Flatten(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.net(x)
        return y


class GeometricTransformModel(nn.Module):
    def __init__(self, transform_res_size):
        super().__init__()
        res_w, res_h = transform_res_size

        # self.feature_extractor = FeatureExtractor()
        self.feature_extractor = nn.Sequential(
            torchvision.models.resnet18(
                pretrained=False,
                progress=False,
                num_classes=32,
            ),
            nn.ReLU(),
        )
        self.st = SpatialUVOffsetTransformer(
            inp=32,
            uv_resolution_shape=(res_w, res_h),
            weight_mult_factor=0.01,
        )

    def forward(self, x):
        self.features = self.feature_extractor(x)
        x = torch.mean(x, dim=1, keepdim=True)
        y_hat = self.st([self.features, x])
        y_hat = y_hat.repeat(1, 3, 1, 1)
        return y_hat

    def criterion(self, y_hat, y):
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
        transform_res_size=(2, 2),
    ).to(DEVICE)
    optim = torch.optim.SGD(model.parameters(), lr=0.00001)

    dl = datamodule.plot_dataloader()
    batch = next(iter(dl))

    x = batch["y"]

    fig = Fig(nr=1, nc=3, ion=True, figsize=(15, 10))
    im = batch["x"].ez.grid(nr=2).channel_last.np
    fig[0].imshow(im)
    fig[0].ax.set_title("Input")

    im = batch["y"].ez.grid(nr=2).channel_last.np
    fig[1].imshow(im)
    fig[1].ax.set_title("Output")
    fig[2].ax.set_title("Predictions")

    for _ in range(100):
        y = model(x)
        loss = model.criterion(y, x)

        optim.zero_grad()
        loss.backward()
        optim.step()

        im = y.ez.grid(nr=2).channel_last.np
        fig[2].imshow(im)
        fig.update()
        print(loss.item())

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sanity_check()
