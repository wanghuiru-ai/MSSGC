from pathlib import Path
from maml.text.dataset import MySentence, MyDataset, MyCorpus


def load_corpus(path: str) -> MyCorpus:
    path = Path(path)
    path_text = path / 'Twitter-2016'
    path_train = path_text / 'train.txt'
    path_dev = path_text / 'dev.txt'
    path_test = path_text / 'test.txt'

    assert path_train.exists()
    assert path_dev.exists()
    assert path_test.exists()

    train = load_dataset(path_train)
    dev = load_dataset(path_dev)
    test = load_dataset(path_test)

    return MyCorpus(train, dev, test)


def load_dataset(path_text: Path) -> MyDataset:
    pairs = []
    sem_dic = {'negative': 0, 'positive': 1, 'neutral': 2}
    with open(str(path_text), encoding='utf-8') as column_file:
        for line in column_file:
            line = line.strip('\n')
            line = line.split('\t')
            text_id, flag, text = line[0], line[1], line[2]
            if flag == 'neutral':
                continue
            pairs.append(MySentence(text, sem_dic[flag]))

    return MyDataset(pairs)


def flag_count(dataset: MyDataset) -> str:
    from collections import Counter
    flags = [pair.flag for pair in dataset]
    counter = Counter(flags)
    sentences = len(dataset)
    n = counter[2]
    positives = counter[1]
    negatives = counter[0]
    return f'{sentences}\t\t{positives}\t\t\t{negatives}\t\t{n}'


if __name__ == "__main__":
    path = '/home/data1/liuyi/MSD/dataset'
    corpus = load_corpus(path)

    print('----------------------------------------------------')
    print('\t\tsentences\tpositives\tnegatives\tneutral')
    print('----------------------------------------------------')
    print('train\t' + flag_count(corpus.train))
    print('dev\t\t' + flag_count(corpus.dev))
    print('test\t' + flag_count(corpus.test))
    print('----------------------------------------------------')
