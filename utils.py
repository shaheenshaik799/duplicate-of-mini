from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loaders(data_dir):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        root=data_dir + "/train",
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=data_dir + "/test",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    return train_loader, test_loader