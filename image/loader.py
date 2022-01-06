from pathlib import Path
from maml.image.dataset import MyImage, MyDataset, MyCorpus


def load_corpus(path: str, split: int = 1) -> MyCorpus:
    path = Path(path)
    path_dataset = path / 'Twitter1269'
    # train_file = 'train_' + str(split) + '.txt'
    train_file = 'train_all.txt'
    # train_file = 'train_sample.txt'
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


def flag_count(dataset: MyDataset) -> str:
    from collections import Counter
    flags = [image.flag for image in dataset]
    counter = Counter(flags)
    sentences = len(dataset)
    positives = counter[1]
    negatives = counter[0]
    return f'{sentences}\t\t{positives}\t\t{negatives}'


if __name__ == "__main__":
    path = '/mnt/E/ly/MSD/dataset'
    corpus = load_corpus(path, split=1)

    print('----------------------------------------------------')
    print('\t\tsentences\tpositives\tnegatives')
    print('----------------------------------------------------')
    print('train\t' + flag_count(corpus.train))
    print('test\t' + flag_count(corpus.test))
    print('----------------------------------------------------')
