import json
import os

import argparse

import numpy


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.loads(f.read())


def get_sst_labels(sst_bert_tokens: dict, phrase_sentiments: dict):
    labels = []
    for sentence in sst_bert_tokens['sst_tokens']:
        text = ""
        for i in range(len(sentence)):
            if i == len(sentence) - 1:
                text += sentence[i]
            else:
                text += sentence[i] + " "
        sentiment = phrase_sentiments[text]
        labels.append(int(round(sentiment)))

    return labels


def get_phrase_sentiments():
    return load_json('datasets_ours/sst/phrase_sentiments.json')


def visualize_token_attrs(tokens, attrs, positive):
    def get_color(attr):
        if attr > 0:
            r = 128 - int(64 * attr)
            g = int(128 * attr) + 127
            b = 128 - int(64 * attr)
        else:
            r = int(-128 * attr) + 127
            g = 128 + int(64 * attr)
            b = 128 + int(64 * attr)
        return r, g, b

    attrs = numpy.array(attrs)

    # if the classified sample is negative, positive attributions will contribute to the negative sentiment
    # so we flip the signs
    if not positive:
        attrs = [a * -1 for a in attrs]

    # normalize attributions for visualization.
    bound = max(abs(attrs.max(axis=0)), abs(attrs.min(axis=0)))
    attrs = attrs / bound
    html_text = ""
    for i, tok in enumerate(tokens):
        r, g, b = get_color(attrs[i])
        html_text += f"<span style='color:rgb({r},{g},{b})'>{tok}</span> "
    return "<html><body>" + html_text + "</body></html>"


def main():
    indices = [int(i) for i in args['indices'].split(',')]
    method_file_dict = load_json(os.path.join(args['attrs_dir'], 'method_file_dict.json'))

    for method, file in method_file_dict.items():
        attrs_all = load_json(args['attrs_file'])

    attrs_all = load_json(args['attrs_file'])
    directory, attrs_file = os.path.split(args['attrs_file'])
    sst_bert_tokens = load_json(os.path.join(directory, 'sst_bert_tokens.json'))
    phrase_sentiments = get_phrase_sentiments()
    labels = get_sst_labels(sst_bert_tokens, phrase_sentiments)

    os.makedirs(os.path.join(args['output_dir'], attrs_file[:-5]))

    for idx in indices:
        tokens = sst_bert_tokens['bert_tokens'][idx]
        attrs = attrs_all[idx]
        label = labels[idx]
        html_text = visualize_token_attrs(tokens, attrs, positive=label == 1)

        with open(os.path.join(args['output_dir'], attrs_file[:-5], f'{idx}.html'), 'w+', encoding='utf-8') as f:
            f.write(html_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', required=True, type=str, help='A comma separated list of indices, e.g. "1,2,3"')
    parser.add_argument('--attrs_dir', required=True, type=str, help='A path to an attributions directory.')
    parser.add_argument('--output_dir', required=False, type=str, default='html_visualisations')
    args = vars(parser.parse_args())
    main()
