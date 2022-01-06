from pathlib import Path
from maml.data.dataset import MySentence, MyImage, MyPair, MyDataset, MyCorpus


def load_corpus(path: str) -> MyCorpus:
    path = Path(path)
    path_text = path / 'text'
    path_train = path_text / 'train.txt'
    path_dev = path_text / 'dev.txt'
    path_test = path_text / 'test.txt'
    path_image = path / 'image'

    assert path_train.exists()
    assert path_dev.exists()
    assert path_test.exists()
    assert path_image.exists()

    train = load_dataset(path_train, path_image)
    dev = load_dataset(path_dev, path_image)
    test = load_dataset(path_test, path_image)

    return MyCorpus(train, dev, test)


def load_dataset(path_text: Path, path_image: Path) -> MyDataset:
    discard_words = ['sarcasm', 'sarcastic', 'irony', 'ironic', '<url>',
                     'reposting', 'joke', 'jokes', 'humour', 'humor', 'exgag']
    pairs = []

    with open(str(path_text), encoding='utf-8') as column_file:
        for line in column_file:
            line = eval(line)
            image_id, text, flag = line[0], line[1], line[-1]
            tokens = text.split()
            for discard_word in discard_words:
                if discard_word in tokens:
                    break
            else:
                pairs.append(MyPair(MySentence(text), MyImage(image_id), flag))

    return MyDataset(pairs, path_image)


def flag_count(dataset: MyDataset) -> str:
    from collections import Counter
    flags = [pair.flag for pair in dataset]
    counter = Counter(flags)
    sentences = len(dataset)
    positives = counter[1]
    negatives = counter[0]
    return f'{sentences}\t\t{positives}\t\t\t{negatives}'


if __name__ == "__main__":
    path = '/home/data1/liuyi/MSD/dataset'
    corpus = load_corpus(path)

    print('--------------------------------------------------')
    print('\t\tsentences\tpositives\tnegatives')
    print('--------------------------------------------------')
    print('train\t' + flag_count(corpus.train))
    print('dev\t\t' + flag_count(corpus.dev))
    print('test\t' + flag_count(corpus.test))
    print('--------------------------------------------------')
