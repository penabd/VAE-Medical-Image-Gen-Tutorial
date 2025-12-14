from torch.utils.data import DataLoader
from torchvision import transforms

from data.chexpert_dataset import CheXpertDataset
from configs import paths


def get_chexpert_dataset(
    csv_path,
    image_root,
    label_name="Pneumonia",
):
    dataset = CheXpertDataset(
        csv_path=csv_path,
        root=image_root,
        label_name=label_name,
    )

    return dataset


def get_chexpert_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def get_chexpert_train_dataloader():
    dataset = get_chexpert_dataset(
        csv_path=paths.CHEXPERT_TRAIN_SUBSET_CSV,
        image_root=paths.CHEXPERT_TRAIN_DIR,
        label_name="Pneumonia",
    )
    dataloader = get_chexpert_dataloader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )
    return dataloader

def get_chexpert_valid_dataloader():
    dataset = get_chexpert_dataset(
        csv_path=paths.CHEXPERT_VALID_CSV,
        image_root=paths.CHEXPERT_VALID_DIR,
        label_name="Pneumonia",
    )
    dataloader = get_chexpert_dataloader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
    )
    return dataloader