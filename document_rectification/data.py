import ez_torch
import kornia
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision
from kornia import augmentation
from torch import nn
from torchvision.datasets.folder import ImageFolder
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor


def map_dl(mapper, dl):
    class DL:
        def __len__(self):
            return len(dl)

        def __iter__(self):
            return (mapper(b) for b in dl)

    return DL()


class ParamCompose:
    def __init__(self, functions):
        self.functions = functions

    def __call__(self, inp, params=None):
        if params is None:
            params = [None] * len(self.functions)

        for f, p in zip(self.functions, params):
            inp = f(inp, p)

        return inp

    def forward_parameters(self, shape):
        params = []
        for f in self.functions:
            p = f.forward_parameters(shape)
            params.append(p)

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
    )
    return augmentor


def get_augmented_dl(path, bs, shuffle):
    dl = get_dl(path, bs=bs, shuffle=shuffle)
    augmentor = get_augmentor()

    def mapper(batch):
        X, _ = batch
        params = augmentor.forward_parameters(X.shape)

        mask = torch.ones_like(X)
        bg = torch.ones_like(X)
        bg[:, 1] = 0

        X = augmentor(X, params)
        mask = augmentor(mask, params)
        transformed_X = X + bg * (1 - mask)

        return {
            "x": transformed_X,
            "y": X,
        }

    return map_dl(mapper, dl)


def get_datamodule(train_bs, val_bs, plot_bs):
    TRAIN_PATH = ".data/dataset/training_data/"
    TEST_PATH = ".data/dataset/testing_data/"

    class DataModule(pl.LightningDataModule):
        def plot_dl(self):
            dl = get_augmented_dl(TRAIN_PATH, bs=plot_bs, shuffle=False)
            example_batch = next(iter(dl))
            return [example_batch]

        def train_dataloader(self):
            return get_augmented_dl(TRAIN_PATH, bs=train_bs, shuffle=True)

        def val_dataloader(self):
            return get_augmented_dl(TEST_PATH, bs=val_bs, shuffle=False)

    return DataModule()


def main():
    dm = get_datamodule(train_bs=32, val_bs=32, plot_bs=16)
    dl = dm.plot_dl()
    batch = next(iter(dl))
    batch["x"].ez.grid(nr=4).imshow(figsize=(8, 8))
    plt.show()


if __name__ == "__main__":
    main()
