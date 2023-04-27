import json
import os

import argparse

import numpy


method_name = {"grads": "grads",
               "grads_x_inputs": "grads x I",
               "ig_20": "ig 20",
               "ig_50": "ig 50",
               "ig_100": "ig 100",
               "sg_20": "sg 20",
               "sg_50": "sg 50",
               "sg_100": "sg 100",
               "sg_20_x_inputs": "sg 20 x I",
               "sg_50_x_inputs": "sg 50 x I",
               "sg_100_x_inputs": "sg 100 x I",
               "ks_100": "ks 100",
               "ks_200": "ks 200",
               "ks_500": "ks 500",
               "relprop": "Chefer"}

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']


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


def add_html(s):
    return '<html><body style="font-family: sans-serif; font-size: 0.75em">' + s + "</body></html>"


def get_pre(letter):
    return f'<pre style="display: inline; tab-size: 2;"><b>({letter})&#9;</b></pre>'


def visualize_token_attrs(tokens, attrs, positive):
    # adapted code from https://github.com/ankurtaly/Integrated-Gradients/blob/master/howto.md
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
        attrs *= -1

    # normalize attributions for visualization.
    bound = max(abs(attrs.max(axis=0)), abs(attrs.min(axis=0)))
    attrs = attrs / bound
    html_text = ""
    for i, tok in enumerate(tokens):
        r, g, b = get_color(attrs[i])
        html_text += f"<span style='color:rgb({r},{g},{b})'><b>{tok}</b></span> "
    return html_text


def main():
    indices = [int(i) for i in args['indices'].split(',')]
    method_file_dict = load_json(os.path.join(args['attrs_dir'], 'method_file_dict.json'))
    sst_bert_tokens = load_json(os.path.join(args['attrs_dir'], 'sst_bert_tokens.json'))
    phrase_sentiments = get_phrase_sentiments()
    labels = get_sst_labels(sst_bert_tokens, phrase_sentiments)
    os.makedirs(args['output_dir'])

    for idx in indices:
        html_text = ''
        for i, (method, file) in enumerate(method_file_dict.items()):
            attrs_all = load_json(os.path.join(args['attrs_dir'], file))

            tokens = sst_bert_tokens['bert_tokens'][idx]
            attrs = attrs_all[idx]
            label = labels[idx]
            html_temp = visualize_token_attrs(tokens, attrs, positive=label == 1)
            html_temp = get_pre(letters[i]) + html_temp + '<br /><br />'
            print(f'{letters[i]}) {method}')
            html_text += html_temp

        with open(os.path.join(args['output_dir'], f'{idx}.html'), 'w+', encoding='utf-8') as f:
            f.write(add_html(html_text))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', required=True, type=str, help='A comma separated list of indices, e.g. "1,2,3"')
    parser.add_argument('--attrs_dir', required=True, type=str, help='A path to an attributions directory.')
    parser.add_argument('--output_dir', required=False, type=str, default='html_visualisations')
    args = vars(parser.parse_args())
    main()
