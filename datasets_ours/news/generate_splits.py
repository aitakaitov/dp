import os
import json
import re


def main():
    with open('classes.json', 'r', encoding='utf-8') as f:
        class_idx_dict = json.loads(f.read())

    used_classes = class_idx_dict.keys()

    train_files = os.listdir('czech_text_document_corpus_v20')
    train_files.remove('dev')
    dev_files = os.listdir(os.path.join('czech_text_document_corpus_v20', 'dev'))

    train_csv = open('train.csv', 'w+', encoding='utf-8')
    for file in train_files:
        classes = file[:-4].split('_')[1:]
        valid_classes = []
        for clss in classes:
            if clss in used_classes:
                valid_classes.append(clss)

        if len(valid_classes) == 0:
            continue

        with open(os.path.join('czech_text_document_corpus_v20', file), 'r', encoding='utf-8') as f:
            text = f.read()
            text = re.sub('\n', ' ', text)

        line = text + '~' + '~'.join(valid_classes)
        train_csv.write(line + '\n')

    train_csv.close()

    dev_csv = open('dev.csv', 'w+', encoding='utf-8')
    for file in dev_files:
        classes = file[:-4].split('_')[1:]
        valid_classes = []
        for clss in classes:
            if clss in used_classes:
                valid_classes.append(clss)

        if len(valid_classes) == 0:
            continue

        with open(os.path.join('czech_text_document_corpus_v20', 'dev', file), 'r', encoding='utf-8') as f:
            text = f.read()
            text = re.sub('\n', ' ', text)

        line = text + '~' + '~'.join(valid_classes)
        dev_csv.write(line + '\n')

    dev_csv.close()

if __name__ == '__main__':
    main()