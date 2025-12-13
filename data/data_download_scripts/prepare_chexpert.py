import os
import pandas as pd

from configs import paths 

def collect_valid_paths(root):
    valid = set()
    for root_dir, _, files in os.walk(root):
        for f in files:
            if f.endswith(".jpg"):
                rel = os.path.relpath(os.path.join(root_dir, f), root)
                valid.add(rel)
    return valid

def prepare_train_csv(
    image_root = paths.DEFAULT_DATA_ROOT,
    csv_in = paths.CHEXPERT_TRAIN_CSV,
    csv_out = paths.CHEXPERT_TRAIN_SUBSET_CSV
):
    print("Collecting valid image paths...")
    valid_paths = collect_valid_paths(image_root)

    print("Loading CSV...")
    df = pd.read_csv(csv_in).fillna(0)

    # Normalize paths
    df["Path"] = df["Path"].str.replace(
        "CheXpert-v1.0-small/train/", "", regex=False
    )

    # Keep frontal views only
    df = df[df["Path"].str.contains("frontal", na=False)]

    # Keep only images that exist on colab
    df = df[df["Path"].isin(valid_paths)]

    df.to_csv(csv_out, index=False)

    print(f"Filtered dataset size: {len(df)}")
    print("Example CSV path:", df["Path"].iloc[0])
    print("Example valid path:", next(iter(valid_paths)))

if __name__ == "__main__":
    prepare_train_csv()
