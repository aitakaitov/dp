import json
import random


def main():
    test_dev_sentences = get_used_sentences()
    train_sentences = get_train_sentences()
    phrases = get_phrases()

    keys = list(phrases.keys())
    random.seed(42)
    random.shuffle(keys)
    phrases = {key: phrases[key] for key in keys}

    train_add_phrases_sent = []
    for phrase, sentiment in phrases.items():
        if len(phrase.split()) < 5:
            continue

        found = False
        for sentence in test_dev_sentences:
            if phrase in sentence:
                found = True
                break

        if found:
            continue

        for sentence in train_sentences:
            if phrase == sentence:
                found = True
                break

        for sentence in train_add_phrases_sent:
            if phrase in sentence:
                found = True
                break

        if not found:
            train_add_phrases_sent.append([phrase, sentiment])

    with open('train.tsv', 'a', encoding='utf-8') as f:
        for phrase, sentiment in train_add_phrases_sent:
            f.write(f'{phrase}\t{sentiment}\n')

    print(len(train_add_phrases_sent))


def get_phrases():
    with open('phrase_sentiments.json', 'r', encoding='utf-8') as f:
        return json.loads(f.read())


def get_train_sentences():
    sentences = []
    with open('train.tsv', 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            sentence, _ = line.strip().split('\t')
            sentences.append(sentence)
    return sentences


def get_used_sentences():
    sentences = []
    with open('dev.tsv', 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            sentence, _ = line.strip().split('\t')
            sentences.append(sentence)

    with open('test.tsv', 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            sentence, _ = line.strip().split('\t')
            sentences.append(sentence)

    return sentences


if __name__ == '__main__':
    main()