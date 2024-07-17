import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from tqdm import tqdm
from termcolor import cprint
import clip
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", image_dir: str = "images") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        self.image_dir = image_dir

        # CLIP用の前処理を定義
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

            # 画像パスのリストを読み込む
            with open(os.path.join(data_dir, f"{split}_image_paths.txt"), "r") as f:
                self.image_paths = f.read().splitlines()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            image_path = os.path.join(self.image_dir, self.image_paths[i])
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image)  # torch.Tensorに変換
            return self.X[i], self.y[i], self.subject_idxs[i], image_tensor
        else:
            return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
