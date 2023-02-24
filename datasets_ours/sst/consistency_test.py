import json

with open('phrase_sentiments.json', 'r', encoding='utf-8') as f:
    phrase_sentiments = json.loads(f.read())

with open('sentences_tokens.json', 'r', encoding='utf-8') as f:
    sentences_tokens = json.loads(f.read())

for sentence, tokens in sentences_tokens:
    for token in tokens:
        if token not in phrase_sentiments.keys():
            print(token)
