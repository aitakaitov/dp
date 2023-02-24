import json

#
#   Creates a join over phrase IDs to connect phrases and their sentiment
#


def create():
    with open('dictionary.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    id_phrase_dict = {}
    for line in lines:
        if line == "":
            continue
        phrase, _id = line.split('|')
        id_phrase_dict[_id.strip()] = phrase

    with open('sentiment_labels.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    phrase_sentiment_dict = {}
    for line in lines[1:]:
        if line == "":
            continue
        _id, sentiment = line.split('|')
        phrase_sentiment_dict[id_phrase_dict[_id]] = float(sentiment.strip())

    with open('phrase_sentiments.json', 'w+', encoding='utf-8') as f:
        f.write(json.dumps(phrase_sentiment_dict))


if __name__ == "__main__":
    create()
