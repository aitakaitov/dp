import json
import re
import argparse


#
#   Uses the datasetSplit, datasetSentences and phrase_sentiment files to create a set of csv files
#   that contain the sentence and the label (labels are binary - 0 or 1)
#

DELTA = 0.1

def clean(sentence):
    # The datasetSentences.txt file has '(' and ')' replaced by character sequences
    # so we sanitize them
    sentence = re.sub('-LRB-', '(', sentence)
    sentence = re.sub('-RRB-', ')', sentence)
    return sentence


def main(args):
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

    dump_to_csv(phrase_sentiments, dataset_sentences_dict[1], 'train.csv', args)
    dump_to_csv(phrase_sentiments, dataset_sentences_dict[2], 'test.csv', args)
    dump_to_csv(phrase_sentiments, dataset_sentences_dict[3], 'dev.csv', args)


def dump_to_csv(phrase_sentiments: dict, sentences: list, file_name: str, args: dict):
    of = open(file_name, 'w+', encoding='utf-8')
    of.write('sentence\tlabel\n')
    for sentence in sentences:
        try:
            sentiment = phrase_sentiments[sentence]
            if not args['keep_neutral']:
                if (0.5 - DELTA) < sentiment < (0.5 + DELTA):
                    continue
        except KeyError:
            continue
        of.write(f'{sentence}\t{0 if sentiment < 0.5 else 1}\n')

    of.close()


def parse_bool(s):
    return s.lower() == 'true'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_neutral', default=False, type=parse_bool)
    args = vars(parser.parse_args())
    main(args)
