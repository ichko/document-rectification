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
            return (mapper(b, i) for i, b in enumerate(dl))

    return DL()


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

        X = augmentor(X, params)
        mask = augmentor(mask, params)
        transformed_X = X + bg * (1 - mask)

        return {
            "x": transformed_X,
            "y": X,
        }

    return map_dl(mapper, cache_dl(dl))


def get_datamodule(train_bs, val_bs, plot_bs, device="cpu"):
    TRAIN_PATH = ".data/dataset/training_data/"
    TEST_PATH = ".data/dataset/testing_data/"

    class DataModule(pl.LightningDataModule):
        def plot_dl(self):
            dl = get_augmented_dl(TRAIN_PATH, bs=plot_bs, shuffle=False, device=device)
            example_batch = next(iter(dl))
            return [example_batch]

        def train_dataloader(self):
            return get_augmented_dl(
                TRAIN_PATH, bs=train_bs, shuffle=True, device=device
            )

        def val_dataloader(self):
            return get_augmented_dl(TEST_PATH, bs=val_bs, shuffle=False, device=device)

    return DataModule()


def main():
    dm = get_datamodule(train_bs=32, val_bs=32, plot_bs=16)
    dl = dm.train_dataloader()

    # i = 0
    # for _ in range(100):
    #     for b in dl:
    #         print(i)
    #         i += 1

    batch = next(iter(dl))
    batch["x"].ez.grid(nr=4).imshow(figsize=(8, 8))
    plt.show()


if __name__ == "__main__":
    main()
