from typing import List
from pathlib import Path
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataPoint:
    def __init__(self):
        self.embedding: Tensor = None


class MyImage(MyDataPoint):
    def __init__(self, image_id: str, flag: int = -1):
        super().__init__()
        self.image_id: str = image_id
        self.data: Tensor = None
        self.flag: int = flag


class MyDataset(Dataset):
    def __init__(self, images: List[MyImage], path: Path):
        self.images: List[MyImage] = images
        self.path: Path = path

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        if image.data is None:
            path_to_image = self.path/f'{image.image_id}'
            image.data = Image.open(path_to_image)
            image.data = image.data.convert('RGB')
            image.data = self.transform(image.data)
        return image


class MyCorpus:
    def __init__(self, train: MyDataset, test: MyDataset):
        self.train: MyDataset = train
        self.test: MyDataset = test
