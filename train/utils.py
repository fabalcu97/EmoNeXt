import torch

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


def repeat_tensor(x):
    return x.repeat(3, 1, 1)


def repeat_crops(crops):
    return torch.stack([crop.repeat(3, 1, 1) for crop in crops])


def crop(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])


def get_image_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.RandomRotation(degrees=20),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            repeat_tensor,
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            repeat_tensor,
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.TenCrop(224),
            crop,
            repeat_crops,
        ]
    )

    return train_transform, validation_transform, test_transform


def get_datasets(configuration):
    train_transform, validation_transform, test_transform = get_image_transforms()

    train_dataset = datasets.ImageFolder(
        configuration.dataset_path + "/train", train_transform)
    validation_dataset = datasets.ImageFolder(
        configuration.dataset_path + "/valid", validation_transform)
    test_dataset = datasets.ImageFolder(
        configuration.dataset_path + "/test", test_transform)

    return train_dataset, validation_dataset, test_dataset


def get_data_loaders(configuration, train_dataset, validation_dataset, test_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=configuration.batch_size,
        shuffle=True,
        num_workers=configuration.num_workers,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, validation_loader, test_loader
