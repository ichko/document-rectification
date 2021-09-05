import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from ez_torch.data import CachedDataset, MapDataset, ParamCompose
from ez_torch.vis import Fig
from kornia.augmentation.augmentation import RandomAffine, RandomPerspective
from torch import nn
from torchvision.datasets.folder import ImageFolder
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor

from document_rectification.common import logger

scale = 0.5
H, W = 1000, 760
H, W = int(H * scale), int(W * scale)


def get_dl(path, bs, shuffle):
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
            RandomAffine(
                degrees=1,
                translate=[0.01, 0.01],
                scale=[0.8, 0.9],
                shear=[-1, 1],
            ),
            RandomPerspective(0.6, p=0.9),
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
            "mask": mask,
        }

    return MapDataset(mapper, CachedDataset(dl, shuffle=shuffle))


class DocumentsDataModule(pl.LightningDataModule):
    H = H
    W = W
    image_size = (H, W)

    def __init__(
        self,
        train_bs,
        val_bs,
        plot_bs,
        shuffle=True,
        device="cpu",
        force_download=False,
    ):
        super().__init__()

        self.shuffle = shuffle
        self.plot_bs = plot_bs
        self.val_bs = val_bs
        self.train_bs = train_bs
        self.device = device

        data_folder = ".data"
        self.TRAIN_PATH = f"{data_folder}/dataset/training_data/"
        self.TEST_PATH = f"{data_folder}/dataset/testing_data/"

        if (
            os.path.exists(data_folder)
            and len(os.listdir(data_folder)) > 0
            and not force_download
        ):
            logger.info(
                "Dataset already download. Use force_download=True to redownload."
            )
        else:
            logger.info("Downloading dataset...")
            os.popen(
                f"poetry run kaggle d download sharmaharsh/form-understanding-noisy-scanned-documentsfunsd -p {data_folder}"
            ).read()
            os.system(
                f"unzip {data_folder}/form-understanding-noisy-scanned-documentsfunsd.zip -d {data_folder}"
            )

    def plot_dataloader(self):
        dl = get_augmented_dl(
            self.TRAIN_PATH,
            bs=self.plot_bs,
            shuffle=False,
            device=self.device,
        )
        example_batch = next(iter(dl))
        """
        (June 15th)
        WARNING: If here I `return [example_batch]` Fig construction will be VERY slow.
                 Specifically plt.subplots (not sure why). It goes from 3s to 23s?!?
                 If you change plot_dataloader -> train_dataloader, the bug goes away.

                 matplotlib#6664 - <https://github.com/matplotlib/matplotlib/issues/6664>
                 Might contain relevant information, but I could not find it.

        (June 16th)
                 This seems to have been fixed by itself. I replaced it with `return [example_batch]`
                 and it works. Not sure why. It might have been related to some caching bug in poetry
                 while updating the ez_torch package.
        """
        return [example_batch]

    def train_dataloader(self):
        return get_augmented_dl(
            self.TRAIN_PATH,
            bs=self.train_bs,
            shuffle=self.shuffle,
            device=self.device,
        )

    def val_dataloader(self):
        return get_augmented_dl(
            self.TEST_PATH,
            bs=self.val_bs,
            shuffle=self.shuffle,
            device=self.device,
        )


def main():
    dm = DocumentsDataModule(train_bs=12, val_bs=32, plot_bs=16, shuffle=False)
    dl = dm.plot_dataloader()
    batch = next(iter(dl))

    fig = Fig(nr=1, nc=2, figsize=(15, 10))
    im = batch["x"].ez.grid(nr=4).channel_last
    fig[0].imshow(im)
    fig[0].ax.set_title("Input")

    im = batch["y"].ez.grid(nr=4).channel_last
    fig[1].imshow(im)
    fig[1].ax.set_title("Output")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
