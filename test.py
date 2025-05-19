# Image Classification
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2

H, W = 32, 32
original_img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)


def repeat_tensor(x):
    return x.repeat(3, 1, 1)


# Image Augmentation
transforms = v2.Compose(
    [
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.Grayscale(),
        v2.Resize(236),
        v2.RandomRotation(degrees=20),
        v2.RandomCrop(224),
    ]
)
transformed_img = transforms(original_img)


def show_images(original, transformed):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original.permute(1, 2, 0).numpy())
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(transformed.permute(1, 2, 0).numpy())
    ax[1].set_title("Transformed Image")
    ax[1].axis("off")
    plt.tight_layout()
    plt.show()


show_images(original_img, transformed_img)
