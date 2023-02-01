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
    with open('datasetSentences.txt', 'r', encoding='utf-8') as f:
        sentences = f.readlines()[1:]
        for i in range(len(sentences)):
            sentences[i] = clean(sentences[i])

    with open('SOStr.txt', 'r', encoding='utf-8') as f:
        tokens = f.readlines()

    processed_pairs = []
    for i in range(len(sentences)):
        split = tokens[i].strip().split('|')
        processed_pairs.append([sentences[i].split('\t')[1], split])

    with open('sentences_tokens.json', 'w+', encoding='utf-8') as f:
        f.write(json.dumps(processed_pairs))


if __name__ == "__main__":
    create()