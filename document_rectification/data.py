import os

import kornia
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from ez_torch.vis import Fig
from torch import nn
from torchvision.datasets.folder import ImageFolder
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor

from document_rectification.common import logger


# Move to eztorch
def map_dl(mapper, dl):
    class DL:
        def __len__(self):
            return len(dl)

        def __iter__(self):
            return (mapper(b, i) for i, b in enumerate(dl))

    return DL()


# Move to eztorch
def cache_dl(dl):
    buffer = []

    class CachedDL:
        def __len__(self):
            return len(dl)

        def __iter__(self):
            if len(buffer) > 0:
                for b in buffer:
                    yield b
            else:
                for b in dl:
                    buffer.append(b)
                    yield b

    return CachedDL()


# Move to eztorch
class ParamCompose(nn.Module):
    def __init__(self, functions):
        super().__init__()
        self.functions = nn.ModuleList(functions)

    def forward(self, inp, params=None):
        if params is None:
            params = [None] * len(self.functions)

        for f, p in zip(self.functions, params):
            inp = f(inp, p)

        return inp

    def forward_parameters(self, shape, device="cpu"):
        params = []
        for f in self.functions:
            p = f.forward_parameters(shape)
            pp = {}
            for k, v in p.items():
                pp[k] = v.to(device)
            params.append(pp)

        return params


def get_dl(path, bs, shuffle):
    scale = 0.5
    H, W = 1000, 762
    H, W = int(H * scale), int(W * scale)
    ds = ImageFolder(
        path,
        transform=Compose(
            [ToTensor(), Normalize((0,), (1,)), Resize((H, W))],
        ),
    )
    dl = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=4,
        drop_last=False,
        persistent_workers=True,
    )
    return dl


def get_augmentor():
    augmentor = ParamCompose(
        [
            kornia.augmentation.RandomAffine(
                degrees=15,
                translate=[0.1, 0.1],
                scale=[0.9, 1.1],
                shear=[-10, 10],
            ),
            kornia.augmentation.RandomPerspective(0.6, p=0.9),
        ]
    ).eval()
    return augmentor


def get_augmented_dl(path, bs, shuffle, device="cpu"):
    dl = get_dl(path, bs=bs, shuffle=shuffle)
    augmentor = get_augmentor()

    def mapper(batch, _idx):
        X, _ = batch
        X = X.to(device)
        params = augmentor.forward_parameters(X.shape, device=device)

        mask = torch.ones_like(X, device=device)
        bg = torch.ones_like(X, device=device)
        bg[:, 1] = 0

        transformed_X = augmentor(X, params)
        mask = augmentor(mask, params)
        transformed_X = transformed_X + bg * (1 - mask)

        return {
            "x": transformed_X,
            "y": X,
        }

    return map_dl(mapper, cache_dl(dl))


def get_datamodule(
    train_bs, val_bs, plot_bs, shuffle=True, device="cpu", force_download=False
):
    data_folder = ".data"
    TRAIN_PATH = f"{data_folder}/dataset/training_data/"
    TEST_PATH = f"{data_folder}/dataset/testing_data/"

    if (
        os.path.exists(data_folder)
        and len(os.listdir(data_folder)) > 0
        and not force_download
    ):
        logger.info("Dataset already download. Use force_download=True to redownload.")
    else:
        logger.info("Downloading dataset...")
        os.popen(
            "poetry run kaggle d download sharmaharsh/form-understanding-noisy-scanned-documentsfunsd -p .data"
        ).read()
        os.system(
            "unzip .data/form-understanding-noisy-scanned-documentsfunsd.zip -d .data"
        )

    class DataModule(pl.LightningDataModule):
        def plot_dl(self):
            dl = get_augmented_dl(TRAIN_PATH, bs=plot_bs, shuffle=False, device=device)
            example_batch = next(iter(dl))
            return [example_batch]

        def train_dataloader(self):
            return get_augmented_dl(
                TRAIN_PATH, bs=train_bs, shuffle=shuffle, device=device
            )

        def val_dataloader(self):
            return get_augmented_dl(
                TEST_PATH, bs=val_bs, shuffle=shuffle, device=device
            )

    return DataModule()


def main():
    dm = get_datamodule(train_bs=12, val_bs=32, plot_bs=16, shuffle=False)
    dl = dm.plot_dl()

    # i = 0
    # for _ in range(100):
    #     for b in dl:
    #         print(i)
    #         i += 1

    batch = next(iter(dl))

    fig = Fig(nr=1, nc=2, figsize=(15, 15))
    fig[0].imshow(batch["x"].ez.grid(nr=2).channel_last)
    fig[0].ax.set_title("Input")

    fig[1].imshow(batch["y"].ez.grid(nr=2).channel_last)
    fig[1].ax.set_title("Output")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
