import json
import random

#
#   Extends the original train split with partial sentences (or phrases) which are not contained in the test and dev
#   splits. The list of phrases that will be added is checked against every considered phrase to limit duplicates.
#   For this reason the phrase dictionary is shuffled, as it is originally ordered with subphrases followed by
#   superphrases - this shuffling prevents the situation where all the subphrases of a sentence are added.
#   Also phrases shorter than 5 tokens are not considered.
#

def main():
    test_dev_sentences = get_used_sentences()
    train_sentences = get_train_sentences()
    phrases = get_phrases()

    keys = list(phrases.keys())

    # shuffle the phrase dictionary
    random.seed(42)
    random.shuffle(keys)
    phrases = {key: phrases[key] for key in keys}

    train_add_phrases_sent = []
    for phrase, sentiment in phrases.items():
        # limit the phrase length
        if len(phrase.split()) < 5:
            continue

        found = False
        # check the phrase against the test and dev sentences to limit overlap between train and test/dev splits
        for sentence in test_dev_sentences:
            if phrase in sentence:
                found = True
                break

        if found:
            continue

        # check the phrase against training phrases to limit partial duplicates
        for sentence in train_sentences:
            if phrase == sentence:
                found = True
                break

        # check the phrase against already added phrases for the same reasons as above
        for sentence in train_add_phrases_sent:
            if phrase in sentence:
                found = True
                break

        if not found:
            train_add_phrases_sent.append([phrase, sentiment])

    with open('train.tsv', 'a', encoding='utf-8') as f:
        for phrase, sentiment in train_add_phrases_sent:
            f.write(f'{phrase}\t{0 if sentiment < 0.5 else 1}\n')

    print(len(train_add_phrases_sent))


def get_phrases():
    with open('phrase_sentiments.json', 'r', encoding='utf-8') as f:
        return json.loads(f.read())


def get_train_sentences():
    sentences = []
    with open('train.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            sentence, _ = line.strip().split('\t')
            sentences.append(sentence)
    return sentences


def get_used_sentences():
    sentences = []
    with open('dev.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            sentence, _ = line.strip().split('\t')
            sentences.append(sentence)

    with open('test.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            sentence, _ = line.strip().split('\t')
            sentences.append(sentence)

    return sentences


if __name__ == '__main__':
    main()