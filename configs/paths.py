import os

# Google Drive Root

# Change if yours doesn't match
GOOGLE_DRIVE_ROOT = "/content/drive/MyDrive/CheXpertSubset"

GOOGLE_DRIVE_CHEXPERT_TRAIN_DIR = os.path.join(GOOGLE_DRIVE_ROOT,
                                                 "CheXpert-v1.0-small/train")
GOOGLE_DRIVE_CHEXPERT_VALID_DIR = os.path.join(GOOGLE_DRIVE_ROOT,
                                                 "CheXpert-v1.0-small/valid")

GOOGLE_DRIVE_CHEXPERT_TRAIN_CSV = os.path.join(GOOGLE_DRIVE_ROOT, "train.csv")
GOOGLE_DRIVE_CHEXPERT_VALID_CSV = os.path.join(GOOGLE_DRIVE_ROOT, "valid.csv")


### Colab Root Directories
# Default root -
DEFAULT_DATA_ROOT = "/content/chexpert"

# Allow override via environment variable - change if yours differs
# from the default
DATA_ROOT = os.getenv("CHEXPERT_ROOT", DEFAULT_DATA_ROOT)

CHEXPERT_TRAIN_DIR = os.path.join(DATA_ROOT, "train")
CHEXPERT_VALID_DIR = os.path.join(DATA_ROOT, "valid")

CHEXPERT_TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
CHEXPERT_TRAIN_SUBSET_CSV = os.path.join(DATA_ROOT, "train.csv")
CHEXPERT_VALID_CSV = os.path.join(DATA_ROOT, "valid.csv")

