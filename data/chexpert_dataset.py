from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import pandas as pd
from torchvision import transforms


class CheXpertDataset(Dataset):
    def __init__(self, csv_path, root, label_name="Pneumonia"):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.fillna(0)

        # Keep only rows with valid image paths
        self.df = self.df[self.df['Path'].notna()]

        self.root = root
        self.label_name = label_name

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        rel_path = row["Path"].replace("CheXpert-v1.0-small/", "")
        img_path = os.path.join(self.root, rel_path)

        # Skip missing images
        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        img = Image.open(img_path).convert("L")
        img = self.transform(img)

        # Labels are -1,0,1 in CheXpert -> convert to {0,1}
        y = torch.tensor([1.0 if row[self.label_name] == 1 else 0.0], dtype=torch.float32)

        return img, y
