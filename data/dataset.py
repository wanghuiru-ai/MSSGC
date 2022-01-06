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
    def __init__(self, text: str):
        super().__init__()
        self.text: str = text


class MyImage(MyDataPoint):
    def __init__(self, image_id: str):
        super().__init__()
        self.image_id: str = image_id
        self.data: Tensor = None


class MyPair(MyDataPoint):
    def __init__(self, sentence: MySentence, image: MyImage, flag: int = -1):
        super().__init__()
        self.sentence: MySentence = sentence
        self.image: MyImage = image
        self.flag: int = flag


class MyDataset(Dataset):
    def __init__(self, pairs: List[MyPair], path: Path):
        self.pairs: List[MyPair] = pairs
        self.path: Path = path

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        image = pair.image
        if image.data is None:
            path_to_image = self.path/f'{image.image_id}.jpg'
            image.data = Image.open(path_to_image)
            image.data = image.data.convert('RGB')
            image.data = self.transform(image.data)
        return pair


class MyCorpus:
    def __init__(self, train: MyDataset, dev: MyDataset, test: MyDataset):
        self.train: MyDataset = train
        self.dev: MyDataset = dev
        self.test: MyDataset = test
