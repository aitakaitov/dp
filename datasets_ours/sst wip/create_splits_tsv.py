import json
import re


def clean(sentence):
    sentence = re.sub('-LRB-', '(', sentence)
    sentence = re.sub('-RRB-', ')', sentence)
    return sentence


def main():
    with open('datasetSplit.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]

    with open('datasetSentences.txt', 'r', encoding='utf-8') as f:
        sentences = f.readlines()[1:]
        for i in range(len(sentences)):
            sentences[i] = clean(sentences[i])

    with open('phrase_sentiments.json', 'r', encoding='utf-8') as f:
        phrase_sentiments = json.loads(f.read())

    dataset_sentences_dict = {1: [], 2: [], 3: []}
    for line in lines:
        sentence_id, dataset_id = line.strip().split(',')
        dataset_sentences_dict[int(dataset_id)].append(sentences[int(sentence_id) - 1].strip().split('\t')[1])

    dump_to_csv(phrase_sentiments, dataset_sentences_dict[1], 'train.tsv')
    dump_to_csv(phrase_sentiments, dataset_sentences_dict[2], 'test.tsv')
    dump_to_csv(phrase_sentiments, dataset_sentences_dict[3], 'dev.tsv')


def dump_to_csv(phrase_sentiments: dict, sentences: list, file_name: str):
    of = open(file_name, 'w+', encoding='utf-8')
    of.write('sentence\tlabel\n')
    for sentence in sentences:
        try:
            sentiment = phrase_sentiments[sentence]
        except KeyError:
            continue
        of.write(f'{sentence}\t{0 if sentiment < 0.5 else 1}\n')

    of.close()


if __name__ == '__main__':
    main()