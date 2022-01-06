from typing import List
from pathlib import Path
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataPoint:
    def __init__(self):
        self.embedding: Tensor = None


class MySentence(MyDataPoint):
    def __init__(self, text: str, flag: int = -1):
        super().__init__()
        self.text: str = text
        self.flag: int = flag


class MyDataset(Dataset):
    def __init__(self, pairs: List[MySentence]):
        self.pairs: List[MySentence] = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        return pair


class MyCorpus:
    def __init__(self, train: MyDataset, dev: MyDataset, test: MyDataset):
        self.train: MyDataset = train
        self.dev: MyDataset = dev
        self.test: MyDataset = test
