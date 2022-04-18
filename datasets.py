import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, unaligned=False):
        super().__init__()

        self.transform = transform
        self.unaligned = unaligned

        self.files_A = sorted(Path(root / "A").glob("*.*"))
        self.files_B = sorted(Path(root / "B").glob("*.*"))

    def __getitem__(self, index):
        filepath_A = self.files_A[index % len(self.files_A)]
        filepath_B = self.files_B[index % len(self.files_B)]

        if self.unaligned:
            filepath_B = random.choice(self.files_B)

        item_A = Image.open(filepath_A)
        item_B = Image.open(filepath_B)

        if self.transform:
            item_A = self.transform(item_A)
            item_B = self.transform(item_B)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
