"""src.utils.data

Plain-English purpose: Original VQ-VAE model and compatibility utilities.

This file is part of the public Space Semantic Compression research demo.
It is documented at module level so researchers, supervisors, and non-specialist
readers can understand where the file fits before reading implementation details.
"""

import os, glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class ImageFolder256(Dataset):
    def __init__(self, root, exts=("jpg","jpeg","png")):
        self.paths = []
        for e in exts:
            self.paths += glob.glob(os.path.join(root, f"**/*.{e}"), recursive=True)
        self.tx = T.Compose([
            T.RandomResizedCrop(256, scale=(0.8,1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)  # -> [-1,1]
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tx(img)

def make_loader(root, batch=64, workers=0, shuffle=True, pin_memory=False):
    ds = ImageFolder256(root)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,      # 0 on Windows avoids shared-mem maps
        pin_memory=pin_memory     # False on Windows
    )

