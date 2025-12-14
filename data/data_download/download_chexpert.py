import os
import subprocess

from configs.paths import DATA_ROOT

KAGGLE_DATASET = "ashery/chexpert"

def check_kaggle():
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        raise RuntimeError("kaggle.json not found. Please upload Kaggle credentials.")

def download_dataset():
    os.makedirs(DATA_ROOT, exist_ok=True)

    if os.path.exists(os.path.join(DATA_ROOT, "CheXpert-v1.0-small")):
        print("CheXpert already downloaded: skipping.")
        return
    else:
        print("CheXpert not found: downloading")

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET],
        check=True
    )
    subprocess.run(["unzip", "chexpert.zip", "-d", DATA_ROOT], check=True)

# if __name__ == "__main__":
#     check_kaggle()
#     download_dataset()
