from google.colab import drive
import subprocess
import os
import time

from configs import paths

'''This script moves the data that is initially downloaded to your Google Drive (for long term storage)
into your Colab data environment for quick, but temporary access.
'''
# Modify if needed
BATCH_SIZE = 50
TOTAL = 2500 

def mount_drive():
    drive.mount("/content/drive")

# used to move data from google drive to colab for better I/O
def rsync_subset(src, dst):
    os.makedirs(dst, exist_ok=True)
    subprocess.run(
        ["rsync", "-avh", "--ignore-errors", src, dst],
        check=False
    )

def batch_copy_patients(src, dst, batch_size, total):
    os.makedirs(dst, exist_ok=True)

    patients = sorted(os.listdir(src))[:total]
    print(f"Found {len(patients)} patient folders. Starting batch copy...")

    for i in range(0, len(patients), batch_size):
        batch = patients[i:i + batch_size]
        print(f"\n=== Copying batch {i//batch_size + 1} ({i} to {i+len(batch)-1}) ===")

        for p in batch:
            src_path = os.path.join(src, p)
            dst_path = os.path.join(dst, p)

            if os.path.exists(dst_path):
                print(f"{p} already exists — skipping")
                continue

            retries = 3
            for attempt in range(1, retries + 1):
                print(f"Copying {p} (attempt {attempt})...")

                result = subprocess.run(
                    ["rsync", "-a", src_path, dst],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if result.returncode == 0:
                    print(f"Finished {p}")
                    break
                else:
                    print(f"Error copying {p}: {result.stderr.strip()}")
                    time.sleep(5)

            if result.returncode != 0:
                print(f"Failed to copy {p} after {retries} attempts.")


if __name__ == "__main__":
    mount_drive()
    rsync_subset(
        paths.GOOGLE_DRIVE_CHEXPERT_VALID_DIR,
        paths.DEFAULT_DATA_ROOT
    )

    rsync_subset(
        paths.GOOGLE_DRIVE_CHEXPERT_VALID_CSV,
        paths.CHEXPERT_VALID_CSV
    )

    rsync_subset(
        paths.GOOGLE_DRIVE_CHEXPERT_TRAIN_CSV,
        paths.CHEXPERT_TRAIN_CSV
    )

    # note that we rsync the training data in batches.
    batch_copy_patients(
        paths.GOOGLE_DRIVE_CHEXPERT_TRAIN_DIR,
        paths.CHEXPERT_TRAIN_DIR, 
        BATCH_SIZE,
        TOTAL)
