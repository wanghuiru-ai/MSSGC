import numpy as np
from numpy.random import choice


def get_orig_train(path: str, dataset_name: str):
    sample_train = []
    with open(path, encoding='utf-8') as column_file:
        # mm dataset
        if dataset_name == 'mm':
            discard_words = ['sarcasm', 'sarcastic', 'irony', 'ironic', '<url>',
                             'reposting', 'joke', 'jokes', 'humour', 'humor', 'exgag']
            for line in column_file:
                line = line.strip('\n')
                orig_line = line
                line = eval(line)
                image_id, text, flag = line[0], line[1], line[-1]
                tokens = text.split()
                for discard_word in discard_words:
                    if discard_word in tokens:
                        break
                else:
                    sample_train.append(orig_line)

        # text dataset
        elif dataset_name == 'text':
            for line in column_file:
                line = line.strip('\n')
                orig_line = line
                line = line.split('\t')
                text_id, flag, text = line[0], line[1], line[2]
                if flag == 'neutral':
                    continue
                else:
                    sample_train.append(orig_line)

        # image dataset
        else:
            for line in column_file:
                line = line.strip('\n')
                sample_train.append(line)
    return sample_train


def sample_train_choice(sample_train: list, size: int = 19816):
    return choice(a=np.array(sample_train), size=size, replace=True)


def list_to_txt(sample_train: [], save_pth: str):
    with open(save_pth, 'w') as f:
        for line in sample_train:
            f.write(line + '\n')


if __name__ == "__main__":
    dataset_path = '/mnt/E/ly/MSD/dataset'

    mm_path = dataset_path + '/text/train.txt'
    sample_mm = sample_train_choice(get_orig_train(mm_path, dataset_name='mm'))
    print(sample_mm)
    print(len(sample_mm))
    mm_save_path = dataset_path + '/text/train_sample.txt'
    list_to_txt(sample_mm, mm_save_path)

    text_path = dataset_path + '/Twitter-2016/train.txt'
    sample_text = sample_train_choice(get_orig_train(text_path, dataset_name='text'))
    print(sample_text)
    print(len(sample_text))
    text_save_path = dataset_path + '/Twitter-2016/train_sample.txt'
    list_to_txt(sample_text, text_save_path)

    image_path = dataset_path + '/Twitter1269/train_1.txt'
    sample_image = sample_train_choice(get_orig_train(image_path, dataset_name='image'))
    print(sample_image)
    print(len(sample_image))
    image_save_path = dataset_path + '/Twitter1269/train_sample.txt'
    list_to_txt(sample_image, image_save_path)

