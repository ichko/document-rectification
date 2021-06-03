import ez_torch
import torch
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


def main():
    documents_path = ".data/dataset/training_data/"
    dl = get_dl(documents_path, bs=16, shuffle=True)
    import matplotlib.pyplot as plt

    X, _ = next(iter(dl))
    iar = X.ez.grid(nr=4).imshow(figsize=(8, 8))
    plt.show()

    print("hello world")


if __name__ == "__main__":
    main()
