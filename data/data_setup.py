from data.data_download.setup_colab import (
    mount_drive,
    rsync_dir,
    rsync_file,
    batch_copy_patients,
)
from configs import paths
from data.data_download.prepare_chexpert import prepare_train_csv

def setup_chexpert_data(
    batch_size=50,
    total=2500,
    force=False
):
    mount_drive()

    print("Syncing validation images...")
    rsync_dir(
        paths.GOOGLE_DRIVE_CHEXPERT_VALID_DIR,
        paths.DEFAULT_DATA_ROOT
    )

    print("Syncing CSVs...")
    rsync_file(
        paths.GOOGLE_DRIVE_CHEXPERT_VALID_CSV,
        paths.CHEXPERT_VALID_CSV
    )
    rsync_file(
        paths.GOOGLE_DRIVE_CHEXPERT_TRAIN_CSV,
        paths.CHEXPERT_TRAIN_CSV
    )

    print("Syncing training images (batched)...")
    batch_copy_patients(
        paths.GOOGLE_DRIVE_CHEXPERT_TRAIN_DIR,
        paths.CHEXPERT_TRAIN_DIR,
        batch_size,
        total
    )

    print("Preparing training CSV subset...")
    prepare_train_csv()

    print("CheXpert data setup complete.")
