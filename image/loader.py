from pathlib import Path
from image.dataset import MyImage, MyDataset, MyCorpus


def load_corpus(path: str, split: int = 1) -> MyCorpus:
    path = Path(path)
    path_dataset = path / 'Twitter1269'
    train_file = 'train_all.txt'
    test_file = 'test_' + str(split) + '.txt'

    path_train = path_dataset / train_file
    path_test = path_dataset / test_file
    path_image = path_dataset / 'twitter1'

    assert path_train.exists()
    assert path_test.exists()

    train = load_dataset(path_train, path_image)
    test = load_dataset(path_test, path_image)

    return MyCorpus(train, test)


def load_dataset(path_dataset: Path, path_image: Path) -> MyDataset:
    images = []
    with open(str(path_dataset), encoding='utf-8') as column_file:
        for line in column_file:
            line = line.strip('\n')
            line = line.split(' ')
            image_id, flag = line
            # print(image_id)
            # print(flag)
            images.append(MyImage(image_id, int(flag)))

    return MyDataset(images, path_image)

