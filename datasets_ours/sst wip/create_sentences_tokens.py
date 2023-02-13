import json
import re

#
#   Creates a file which contains sentences and their respective tokens
#

def clean(sentence):
    # The datasetSentences.txt file has '(' and ')' replaced by character sequences
    # so we sanitize them
    sentence = re.sub('-LRB-', '(', sentence)
    sentence = re.sub('-RRB-', ')', sentence)
    return sentence


def create():
    with open('test.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]

    with open('datasetSentences.txt', 'r', encoding='utf-8') as f:
        sentence_id_dict = {}
        sentences = f.readlines()[1:]
        for i in range(len(sentences)):
            sentence_id_dict[clean(sentences[i]).strip().split('\t')[1]] = i + 1
            sentences[i] = clean(sentences[i])

    with open('SOStr.txt', 'r', encoding='utf-8') as f:
        tokens = f.readlines()

    processed_pairs = []
    for line in lines:
        sentence = line.strip().split('\t')[0]
        _id = sentence_id_dict[sentence]
        split = tokens[_id - 1].strip().split('|')
        processed_pairs.append([sentence, split])

    with open('sentences_tokens_test.json', 'w+', encoding='utf-8') as f:
        f.write(json.dumps(processed_pairs))


if __name__ == "__main__":
    create()