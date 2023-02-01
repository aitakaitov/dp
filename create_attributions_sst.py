import torch
import json
from transformers import AutoTokenizer
from attribution_methods_custom import gradient_attributions, ig_attributions, sg_attributions
from models.bert_512 import BertSequenceClassifierSST
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator

import os
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

UNSURE_PREDICTION_DELTA = 0.1

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='output_sst_attrs', help='Attributions output directory')
parser.add_argument('--model_path', required=True, help='Trained model')
parser.add_argument('--baselines_dir', required=True, help='Directory with baseline examples')

args = parser.parse_args()

OUTPUT_DIR = args.output_dir
MODEL_PATH = args.model_path
BASELINES_DIR = args.baselines_dir

CERTAIN_DIR = 'certain'
UNSURE_DIR = 'unsure'

try:
    os.mkdir(OUTPUT_DIR)
    os.mkdir(os.path.join(OUTPUT_DIR, CERTAIN_DIR))
    os.mkdir(os.path.join(OUTPUT_DIR, UNSURE_DIR))
except OSError:
    pass

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = BertSequenceClassifierSST.from_pretrained(MODEL_PATH, local_files_only=True)
model = model.to(device)
model.eval()
embeddings = model.bert.base_model.embeddings.word_embeddings.weight.data

relprop_explainer = Generator(model)

method_file_dict = {
    'grads': 'gradient_attrs_custom.json',
    'grads_x_inputs':  'gradients_x_inputs_attrs_custom.json',
    'ig_20':  'ig_20_attrs_custom.json',
    'ig_50':  'ig_50_attrs_custom.json',
    'ig_100':  'ig_100_attrs_custom.json',
    'ig_200':  'ig_200_attrs_custom.json',
    'sg_20':  'sg_20_attrs_custom.json',
    'sg_50':  'sg_50_attrs_custom.json',
    'sg_100':  'sg_100_attrs_custom.json',
    'sg_200':  'sg_200_attrs_custom_bk.json',
    'sg_20_x_inputs':  'sg_20_x_inputs_attrs_custom.json',
    'sg_50_x_inputs':  'sg_50_x_inputs_attrs_custom.json',
    'sg_100_x_inputs':  'sg_100_x_inputs_attrs_custom.json',
    'sg_200_x_inputs':  'sg_200_x_inputs_attrs_custom.json',
    'relprop':  'relprop_attrs.json'
}

#   -----------------------------------------------------------------------------------------------


def get_sentences_tokens_and_phrase_sentiments():
    with open('datasets_ours/sst/sentences_tokens_orig_test_no_neutral.json', 'r', encoding='utf-8') as f:
        sentences_tokens = json.loads(f.read())

    with open('datasets_ours/sst/phrase_sentiments.json', 'r', encoding='utf-8') as f:
        phrase_sentiments = json.loads(f.read())

    sentences = []
    tokens = []
    for s, t in sentences_tokens:
        sentences.append(s)
        tokens.append(t)

    return sentences, tokens, phrase_sentiments


def get_sentence_sentiments(tokens, phrase_sentiments):
    output = []
    for token in tokens:
        output.append(phrase_sentiments[token])
    return output


def format_attrs(attrs, sentence):
    tokenized = tokenizer(sentence)

    if len(attrs.shape) == 2 and attrs.shape[0] == 1:
        attrs = torch.squeeze(attrs)

    attrs_list = attrs.tolist()
    return attrs_list[1:len(tokenized.data['input_ids']) - 1]  # leave out cls and sep


def prepare_embeds_and_att_mask(sentence):
    encoded = tokenizer(sentence, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    attention_mask = encoded.data['attention_mask'].to(device)
    input_ids = torch.squeeze(encoded.data['input_ids']).to(device)
    input_embeds = torch.unsqueeze(torch.index_select(embeddings, 0, input_ids), 0).requires_grad_(True).to(device)
    input_ids.to('cpu')

    return input_embeds, attention_mask


def prepare_input_ids_and_attention_mask(sentence):
    encoded = tokenizer(sentence, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    attention_mask = encoded.data['attention_mask'].to(device)
    input_ids = encoded.data['input_ids'].to(device)

    return input_ids, attention_mask


def create_neutral_baseline(sentence):
    # we have baselines precomputed for all the lengths of test sentences, to that's ok
    tokenized = tokenizer(sentence)
    length = len(tokenized.data['input_ids'])

    return torch.load(BASELINES_DIR + '/' + str(length) + '.pt').to(device)


#   -----------------------------------------------------------------------------------------------

def create_gradient_attributions(sentences, target_indices, target_dir=CERTAIN_DIR):
    attrs = []
    for sentence, target_idx in zip(sentences, target_indices):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attr = gradient_attributions(input_embeds, attention_mask, target_idx, model)
        attr = torch.squeeze(attr)
        attrs.append(format_attrs(attr, sentence))

    with open(str(os.path.join(OUTPUT_DIR, target_dir, method_file_dict['grads'])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))

    attrs = []
    for sentence, target_idx in zip(sentences, target_indices):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attr = gradient_attributions(input_embeds, attention_mask, target_idx, model, True)
        attr = torch.squeeze(attr)
        attrs.append(format_attrs(attr, sentence))

    with open(str(os.path.join(OUTPUT_DIR, target_dir, method_file_dict['grads_x_inputs'])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))


def _do_ig(sentences, target_indices, steps, file, target_dir):
    attrs = []
    for sentence, target_idx in zip(sentences, target_indices):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        baseline = create_neutral_baseline(sentence)
        attr = ig_attributions(input_embeds, attention_mask, target_idx, baseline, model, steps)
        attr = torch.squeeze(attr)
        attrs.append(format_attrs(attr, sentence))

    with open(str(os.path.join(OUTPUT_DIR, target_dir, method_file_dict[file])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))


def create_ig_attributions(sentences, target_indices, target_dir=CERTAIN_DIR):
    _do_ig(sentences, target_indices, 20, 'ig_20', target_dir)
    _do_ig(sentences, target_indices, 50, 'ig_50', target_dir)
    _do_ig(sentences, target_indices, 100, 'ig_100', target_dir)
    _do_ig(sentences, target_indices, 200, 'ig_200', target_dir)


def _do_sg(sentences, target_indices, samples, file, target_dir):
    attrs = []
    attrs_x_inputs = []
    for sentence, target_idx in zip(sentences, target_indices):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attr = sg_attributions(input_embeds, attention_mask, target_idx, model, samples, 0.2)
        attr_x_input = attr.to(device) * input_embeds
        attr_x_input = torch.squeeze(attr_x_input)
        attr = torch.squeeze(attr)
        attrs.append(format_attrs(attr, sentence))
        attrs_x_inputs.append(format_attrs(attr_x_input, sentence))

    with open(str(os.path.join(OUTPUT_DIR, target_dir, method_file_dict[file])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))

    with open(str(os.path.join(OUTPUT_DIR, target_dir, method_file_dict[file + '_x_inputs'])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs_x_inputs))


def create_smoothgrad_attributions(sentences, target_indices, target_dir=CERTAIN_DIR):
    _do_sg(sentences, target_indices, 20, 'sg_20', target_dir)
    _do_sg(sentences, target_indices, 50, 'sg_50', target_dir)
    _do_sg(sentences, target_indices, 100, 'sg_100', target_dir)
    _do_sg(sentences, target_indices, 200, 'sg_200', target_dir)


def create_relprop_attributions(sentences, target_indices, target_dir=CERTAIN_DIR):
    attrs = []
    for sentence, target_idx in zip(sentences, target_indices):
        input_ids, attention_mask = prepare_input_ids_and_attention_mask(sentence)
        inputs_embeds, _ = prepare_embeds_and_att_mask(sentence)
        res = relprop_explainer.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index=target_idx)
        attrs.append(format_attrs(res, sentence))

    with open(str(os.path.join(OUTPUT_DIR, target_dir, method_file_dict['relprop'])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))


#   -----------------------------------------------------------------------------------------------


def main():
    sentences, tokens, phrase_sentiments = get_sentences_tokens_and_phrase_sentiments()

    # for correct and sure predictions
    bert_tokens_correct = []
    sst_tokens_correct = []
    correct_pred_indices = []
    correct_pred_sentences = []

    # for correct but unsure predictions
    bert_tokens_unsure = []
    sst_tokens_unsure = []
    unsure_pred_indices = []
    unsure_pred_sentences = []

    for sentence, tokens in zip(sentences, tokens):
        # create_neutral_baseline(sentence)
        # first classify the sample
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        res = model(inputs_embeds=input_embeds, attention_mask=attention_mask, return_logits=False, inputs_embeds_in_input_ids=False)
        top_idx = int(torch.argmax(res, dim=-1))
        # compare it to the true sentiment - on mismatch ignore, on correct prediction save
        true_sentiment = phrase_sentiments[sentence]
        if int(round(true_sentiment)) != top_idx:
            continue
        elif (0.5 - UNSURE_PREDICTION_DELTA) < float(res[0, top_idx]) < (0.5 + UNSURE_PREDICTION_DELTA):
            # if the prediction is uncertain, save it separately
            unsure_pred_indices.append(top_idx)
            bert_tokens_unsure.append(tokenizer.tokenize(sentence))
            sst_tokens_unsure.append(tokens)
            unsure_pred_sentences.append(sentence)
        else:
            # save certain predictions
            correct_pred_indices.append(top_idx)
            bert_tokens_correct.append(tokenizer.tokenize(sentence))
            sst_tokens_correct.append(tokens)
            correct_pred_sentences.append(sentence)

    # dump the tokens and predictions
    with open(OUTPUT_DIR + '/sst_bert_tokens.json', 'w+', encoding='utf-8') as f:
        f.write(json.dumps({'bert_tokens': bert_tokens_correct, 'sst_tokens': sst_tokens_correct}))

    with open(OUTPUT_DIR + '/method_file_dict_custom.json', 'w+', encoding='utf-8') as f:
        f.write(json.dumps(method_file_dict))

    # create attributions for the correctly predicted and certain
    create_gradient_attributions(correct_pred_sentences, correct_pred_indices)
    create_smoothgrad_attributions(correct_pred_sentences, correct_pred_indices)
    create_ig_attributions(correct_pred_sentences, correct_pred_indices)
    create_relprop_attributions(correct_pred_sentences, correct_pred_indices)

    # create attributions for the correctly predicted but uncertain
    create_gradient_attributions(correct_pred_sentences, correct_pred_indices, UNSURE_DIR)
    create_smoothgrad_attributions(correct_pred_sentences, correct_pred_indices, UNSURE_DIR)
    create_ig_attributions(correct_pred_sentences, correct_pred_indices, UNSURE_DIR)
    create_relprop_attributions(correct_pred_sentences, correct_pred_indices, UNSURE_DIR)


if __name__ == '__main__':
    main()
