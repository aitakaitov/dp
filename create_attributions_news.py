import argparse

import torch
import json

from transformers import AutoTokenizer, AutoModel, AutoConfig
from attribution_methods_custom import gradient_attributions, ig_attributions, sg_attributions
from models.bert_512 import BertSequenceClassifierNews, ElectraSequenceClassifierNews, RobertaSequenceClassifierNews
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from utils.list_utils import count_rec

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CERTAIN_DIR = 'certain'
UNSURE_DIR = 'unsure'

UNSURE_PREDICTION_DELTA = 0.1

method_file_dict = {
    'grads': 'gradient_attrs_custom.json',
    'grads_x_inputs':  'gradients_x_inputs_attrs_custom.json',
    'ig_20':  'ig_20_attrs_custom.json',
    'ig_50':  'ig_50_attrs_custom.json',
    'ig_100':  'ig_100_attrs_custom.json',
    'sg_20':  'sg_20_attrs_custom.json',
    'sg_50':  'sg_50_attrs_custom.json',
    'sg_100':  'sg_100_attrs_custom.json',
    'sg_20_x_inputs':  'sg_20_x_inputs_attrs_custom.json',
    'sg_50_x_inputs':  'sg_50_x_inputs_attrs_custom.json',
    'sg_100_x_inputs':  'sg_100_x_inputs_attrs_custom.json',
    'relprop':  'relprop_attrs.json'
}

#   -----------------------------------------------------------------------------------------------


def parse_csv_line(line: str):
    split = line.strip('\n').split('~')
    text = split[0]
    classes = split[1:]
    return text, classes


def get_data():
    with open('datasets_ours/news/classes.json', 'r', encoding='utf-8') as f:
        class_dict = json.loads(f.read())

    with open('datasets_ours/news/test.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    samples = []
    labels = []
    for line in lines:
        text, classes = parse_csv_line(line)
        classes = [class_dict[clss] for clss in classes]
        samples.append(text)
        labels.append(classes)

    return samples, labels


def format_attrs(attrs, sentence):
    tokenized = tokenizer(sentence)

    if len(attrs.shape) == 2 and attrs.shape[0] == 1:
        attrs = torch.squeeze(attrs)

    attrs_list = attrs.tolist()
    return attrs_list[1:len(tokenized.data['input_ids']) - 1]  # leave out cls and sep


def prepare_embeds_and_att_mask(sentence):
    encoded = tokenizer(sentence, max_length=512, truncation=True, return_tensors='pt')
    attention_mask = encoded.data['attention_mask'].to(device)
    input_embeds = torch.unsqueeze(torch.index_select(embeddings, 0, torch.squeeze(encoded.data['input_ids']).to(device)), 0).requires_grad_(True).to(device)

    return input_embeds, attention_mask


def prepare_input_ids_and_attention_mask(sentence):
    encoded = tokenizer(sentence, max_length=512, truncation=True, return_tensors='pt')
    attention_mask = encoded.data['attention_mask'].to(device)
    input_ids = encoded.data['input_ids'].to(device)

    return input_ids, attention_mask


def create_neutral_baseline(sentence, pad_token):
    length = len(tokenizer.tokenize(sentence))
    baseline_text = " ".join([pad_token for _ in range(length)])
    inputs_embeds, attention_mask = prepare_embeds_and_att_mask(baseline_text)
    return inputs_embeds


#   -----------------------------------------------------------------------------------------------

def create_gradient_attributions(sentences, target_indices_list, target_dir=CERTAIN_DIR):
    if args['part'] != 'g_sg20-50' and args['part'] != 'all':
        return

    file = open(os.path.join(args['output_dir'], target_dir, method_file_dict['grads']), 'w+', encoding='utf-8')
    file.write('[\n')

    for sentence, target_indices in zip(sentences, target_indices_list):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attrs_temp = []
        for target_idx in target_indices:
            attr = gradient_attributions(input_embeds, attention_mask, target_idx, model)
            attr = torch.squeeze(attr)
            attrs_temp.append(format_attrs(attr, sentence))

        file.write(json.dumps(attrs_temp) + ',')

    file.seek(file.tell() - 1)
    file.write('\n]')
    file.close()

    file = open(os.path.join(args['output_dir'], target_dir, method_file_dict['grads_x_inputs']), 'w+', encoding='utf-8')
    file.write('[\n')

    for sentence, target_indices in zip(sentences, target_indices_list):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attrs_temp = []
        for target_idx in target_indices:
            attr = gradient_attributions(input_embeds, attention_mask, target_idx, model, True)
            attr = torch.squeeze(attr)
            attrs_temp.append(format_attrs(attr, sentence))

        file.write(json.dumps(attrs_temp) + ',')

    file.seek(file.tell() - 1)
    file.write('\n]')
    file.close()


def _do_ig(sentences, target_indices_list, steps, file, target_dir):
    file = open(os.path.join(args['output_dir'], target_dir, method_file_dict[file]), 'w+', encoding='utf-8')
    file.write('[\n')

    for sentence, target_indices in zip(sentences, target_indices_list):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attrs_temp = []
        for target_idx in target_indices:
            baseline = create_neutral_baseline(sentence, tokenizer.pad_token)
            attr = ig_attributions(input_embeds, attention_mask, target_idx, baseline, model, steps)
            attr = torch.squeeze(attr)
            attrs_temp.append(format_attrs(attr, sentence))

        file.write(json.dumps(attrs_temp) + ',')

    file.seek(file.tell() - 1)
    file.write('\n]')
    file.close()


def create_ig_attributions(sentences, target_indices, target_dir=CERTAIN_DIR):
    if args['part'] == 'all':
        _do_ig(sentences, target_indices, 20, 'ig_20', target_dir)
        _do_ig(sentences, target_indices, 50, 'ig_50', target_dir)
        _do_ig(sentences, target_indices, 100, 'ig_100', target_dir)
    elif args['part'] == 'rp_ig20-50':
        _do_ig(sentences, target_indices, 20, 'ig_20', target_dir)
        _do_ig(sentences, target_indices, 50, 'ig_50', target_dir)
    elif args['part'] == 'ig100':
        _do_ig(sentences, target_indices, 100, 'ig_100', target_dir)


def _do_sg(sentences, target_indices_list, samples, file, target_dir):
    f = open(os.path.join(args['output_dir'], target_dir, method_file_dict[file]), 'w+', encoding='utf-8')
    f.write('[\n')

    f_x_inputs = open(os.path.join(args['output_dir'], target_dir, method_file_dict[file + '_x_inputs']), 'w+', encoding='utf-8')
    f_x_inputs.write('[\n')

    for sentence, target_indices in zip(sentences, target_indices_list):
        inputs_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        temp_attrs = []
        temp_attrs_x_inputs = []
        for target_idx in target_indices:
            attr = sg_attributions(inputs_embeds, attention_mask, target_idx, model, samples)
            attr_x_input = attr.to(device) * inputs_embeds
            attr_x_input = torch.squeeze(attr_x_input)
            attr = torch.squeeze(attr)
            temp_attrs.append(format_attrs(attr, sentence))
            temp_attrs_x_inputs.append(format_attrs(attr_x_input, sentence))

        f.write(json.dumps(temp_attrs) + ',')
        f_x_inputs.write(json.dumps(temp_attrs_x_inputs) + ',')

    f.seek(f.tell() - 1)
    f.write('\n]')
    f.close()

    f_x_inputs.seek(f_x_inputs.tell() - 1)
    f_x_inputs.write('\n]')
    f_x_inputs.close()


def create_smoothgrad_attributions(sentences, target_indices, target_dir=CERTAIN_DIR):
    if args['part'] == 'all':
        _do_sg(sentences, target_indices, 20, 'sg_20', target_dir)
        _do_sg(sentences, target_indices, 50, 'sg_50', target_dir)
        _do_sg(sentences, target_indices, 100, 'sg_100', target_dir)
    elif args['part'] == 'g_sg20-50':
        _do_sg(sentences, target_indices, 20, 'sg_20', target_dir)
        _do_sg(sentences, target_indices, 50, 'sg_50', target_dir)
    elif args['part'] == 'sg100':
        _do_sg(sentences, target_indices, 100, 'sg_100', target_dir)


def create_relprop_attributions(sentences, target_indices_list, target_dir=CERTAIN_DIR):
    if args['part'] != 'rp_ig20-50' and args['part'] != 'all':
        return

    f = open(os.path.join(args['output_dir'], target_dir, method_file_dict['relprop']), 'w+', encoding='utf-8')
    f.write('[\n')

    for sentence, target_indices in zip(sentences, target_indices_list):
        input_ids, attention_mask = prepare_input_ids_and_attention_mask(sentence)
        inputs_embeds, _ = prepare_embeds_and_att_mask(sentence)
        temp_attrs = []
        for target_idx in target_indices:
            res = relprop_explainer.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index=target_idx)
            temp_attrs.append(format_attrs(res, sentence))

        f.write(json.dumps(temp_attrs) + ',')

    f.seek(f.tell() - 1)
    f.write('\n]')
    f.close()

#   -----------------------------------------------------------------------------------------------


def main():
    documents, labels = get_data()

    # lists for certain predictions
    bert_tokens_certain = []
    target_indices_certain = []
    valid_documents_certain = []

    # lists for unsure predictions
    bert_tokens_unsure = []
    target_indices_unsure = []
    valid_documents_unsure = []

    # list for counting the actually valid labels
    labels_short_enough = []

    for document, label in zip(documents[:100], labels[:100]):
        # check the length - no longer than 512 tokens
        if len(tokenizer.tokenize(document)) + 2 > 512:
            continue
        else:
            labels_short_enough.append(label)

        # first classify the sample
        input_embeds, attention_mask = prepare_embeds_and_att_mask(document)
        res = model(inputs_embeds=input_embeds, attention_mask=attention_mask, inputs_embeds_in_input_ids=False)
        res = list(torch.squeeze(res))

        # check which labels we have predicted correctly
        target_certain = []
        target_unsure = []
        for i in range(len(res)):
            if i in label and res[i] >= 0.5 + UNSURE_PREDICTION_DELTA:
                target_certain.append(i)
            elif i in label:
                target_unsure.append(i)

        # save the correct and certain predictions
        if len(target_certain) != 0:
            target_indices_certain.append(target_certain)
            bert_tokens_certain.append(tokenizer.tokenize(document))
            valid_documents_certain.append(document)

        # save the correct but unsure predictions
        if len(target_unsure) != 0:
            target_indices_unsure.append(target_unsure)
            bert_tokens_unsure.append(tokenizer.tokenize(document))
            valid_documents_unsure.append(document)

    # dump the tokens
    with open(os.path.join(args['output_dir'], CERTAIN_DIR, 'bert_tokens.json'), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(bert_tokens_certain))
    with open(os.path.join(args['output_dir'], UNSURE_DIR, 'bert_tokens.json'), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(bert_tokens_unsure))

    # dump the indices
    with open(os.path.join(args['output_dir'], CERTAIN_DIR, 'target_indices.json'), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(target_indices_certain))
    with open(os.path.join(args['output_dir'], UNSURE_DIR, 'target_indices.json'), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(target_indices_unsure))

    do_relprop = any(['Bert' in arch for arch in model.config.architectures])

    with open(os.path.join(args['output_dir'], 'method_file_dict.json'), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(method_file_dict))

    create_gradient_attributions(valid_documents_certain, target_indices_certain)
    create_smoothgrad_attributions(valid_documents_certain, target_indices_certain)
    create_ig_attributions(valid_documents_certain, target_indices_certain)

    create_gradient_attributions(valid_documents_unsure, target_indices_unsure, UNSURE_DIR)
    create_smoothgrad_attributions(valid_documents_unsure, target_indices_unsure, UNSURE_DIR)
    create_ig_attributions(valid_documents_unsure, target_indices_unsure, UNSURE_DIR)

    if do_relprop:
        create_relprop_attributions(valid_documents_certain, target_indices_certain)
        create_relprop_attributions(valid_documents_unsure, target_indices_unsure, UNSURE_DIR)
    else:
        method_file_dict.pop('relprop')

    # print report
    print(f'Total labels: {count_rec(labels_short_enough)}')
    print(f'Correctly predicted labels: {count_rec(target_indices_certain) + count_rec(target_indices_unsure)}')
    print(f'Labels predicted certainly: {count_rec(target_indices_certain)}')
    print(f'Labels predicted unsurely: {count_rec(target_indices_unsure)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output_news_attrs', help='Attributions output directory')
    parser.add_argument('--model_path', required=True, help='Trained model')
    parser.add_argument('--part', required=False, default='all',
                        help='Which split to compute - one of [g_sg20-50, sg100, sg200, rp_ig20-50, ig100, ig200, all]')

    args = vars(parser.parse_args())

    try:
        os.mkdir(args['output_dir'])
    except OSError:
        pass

    config = AutoConfig.from_pretrained(args['model_path'])

    if any(['Electra' in arch for arch in config.architectures]):
        if 'small-e-czech' in config.name_or_path:
            # special case for our pretrained models as the tokenizer doesn't
            # load properly
            tokenizer = AutoTokenizer.from_pretrained('Seznam/small-e-czech')
        else:
            tokenizer = AutoTokenizer.from_pretrained(args['model_path'])
        model = ElectraSequenceClassifierNews.from_pretrained(args['model_path'])
        embeddings = model.electra.base_model.embeddings.word_embeddings.weight.data
    elif any(['Bert' in arch for arch in config.architectures]):
        tokenizer = AutoTokenizer.from_pretrained(args['model_path'])
        model = BertSequenceClassifierNews.from_pretrained(args['model_path'])
        embeddings = model.bert.base_model.embeddings.word_embeddings.weight.data
    elif any(['Roberta' in arch for arch in config.architectures]):
        tokenizer = AutoTokenizer.from_pretrained(args['model_path'])
        model = RobertaSequenceClassifierNews.from_pretrained(args['model_path'])
        embeddings = model.bert.base_model.embeddings.word_embeddings.weight.data
    else:
        raise RuntimeError(f'Architectures {config.architectures} not supported')

    ids = tokenizer.encode(tokenizer.pad_token_id)
    pad_token_index = tokenizer.pad_token_id
    cls_token_index = tokenizer.cls_token_id
    sep_token_index = tokenizer.sep_token_id

    embeddings = embeddings.to(device)
    model = model.to(device)

    model.eval()
    relprop_explainer = Generator(model)

    main()
