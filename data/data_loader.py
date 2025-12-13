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

def get_chexpert_train():
    return get_chexpert_dataset(
        csv_path=CHEXPERT_TRAIN_CSV,
        image_root=CHEXPERT_TRAIN_DIR,
    )

def get_chexpert_valid():
    return get_chexpert_dataset(
        csv_path=CHEXPERT_VALID_CSV,
        image_root=CHEXPERT_VALID_DIR,
    )