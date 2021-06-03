import ez_torch
import kornia
import matplotlib.pyplot as plt
import torch
from kornia import augmentation
from torch import nn
from torchvision.datasets.folder import ImageFolder
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor


def get_dataset(path):
    scale = 0.5
    H, W = 1000, 762
    H, W = int(H * scale), int(W * scale)
    ds = ImageFolder(
        path,
        transform=Compose(
            [ToTensor(), Normalize((0,), (1,)), Resize((H, W))],
        ),
    )
    return ds


def get_dl(path, bs, shuffle):
    ds = get_dataset(path)
    dl = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=bs,
        shuffle=shuffle,
    )
    return dl


def get_augmentor():
    # augmentor = Compose(
    #     [
    #         kornia.augmentation.RandomAffine(
    #             degrees=30,
    #             translate=[0.1, 0.1],
    #             scale=[0.9, 1.1],
    #             shear=[-10, 10],
    #         ),
    #         kornia.augmentation.RandomPerspective(0.6, p=0.5),
    #     ]
    # )
    augmentor = kornia.augmentation.RandomPerspective(0.6, p=0.9)
    return augmentor


def main():
    documents_path = ".data/dataset/training_data/"
    dl = get_dl(documents_path, bs=16, shuffle=True)
    augmentor = get_augmentor()

    X, _ = next(iter(dl))

    params = augmentor.forward_parameters(X.shape)

    mask = torch.ones_like(X)
    bg = torch.ones_like(X)
    bg[:, 1] = 0

    X = augmentor(X, params)
    mask = augmentor(mask, params)

    (X + bg * (1 - mask)).ez.grid(nr=4).imshow(figsize=(8, 8))
    plt.show()


if __name__ == "__main__":
    main()
