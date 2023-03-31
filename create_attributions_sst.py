import math

import torch
import json

import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig
from attribution_methods_custom import gradient_attributions, ig_attributions, sg_attributions, kernel_shap_attributions
from models.bert_512 import BertForSequenceClassificationChefer
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
import utils.baselines

import os
import argparse

from utils.check_relprop import is_relprop_possible

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# How far from 0.5 we can be for the prediction to be uncertain
UNSURE_PREDICTION_DELTA = 0.1

# directory for certain and correct predictions
CERTAIN_DIR = 'certain'
# directory for correct but uncertain prediction
UNSURE_DIR = 'unsure'

method_file_dict = {
    'grads': 'gradient_attrs.json',
    'grads_x_inputs':  'gradients_x_inputs_attrs.json',
    'ig_20':  'ig_20_attrs.json',
    'ig_50':  'ig_50_attrs.json',
    'ig_100':  'ig_100_attrs.json',
    'sg_20':  'sg_20_attrs.json',
    'sg_50':  'sg_50_attrs.json',
    'sg_100':  'sg_100_attrs.json',
    'sg_20_x_inputs':  'sg_20_x_inputs_attrs.json',
    'sg_50_x_inputs':  'sg_50_x_inputs_attrs.json',
    'sg_100_x_inputs':  'sg_100_x_inputs_attrs.json',
    'ks_100': 'ks_100_attrs.json',
    'ks_200': 'ks_200_attrs.json',
    'ks_500': 'ks_500_attrs.json',
    'relprop':  'relprop_attrs.json'
}

sg_noise_configs = {
    'bert-base-cased': 0.05,
    'bert-medium': 0.05,
    'bert-small': 0.05,
    'bert-mini': 0.05
}

sg_x_i_noise_configs = {
    'bert-base-cased': 0.05,
    'bert-medium': 0.05,
    'bert-small': 0.05,
    'bert-mini': 0.15
}

ig_baseline_configs = {
    'bert-base-cased': 'zero',
    'bert-medium': 'zero',
    'bert-small': 'avg',
    'bert-mini': 'pad'
}

ks_baseline_configs = {
    'bert-base-cased': 'pad',
    'bert-medium': 'mask',
    'bert-small': 'unk',
    'bert-mini': 'pad'
}


def get_sg_noise(model_path):
    for k in sg_noise_configs.keys():
        if k in model_path:
            return sg_noise_configs[k]

    raise RuntimeError(f'Model {model_path} is not supported with --use_predefined_hp set to True')


def get_sg_x_i_noise(model_path):
    for k in sg_x_i_noise_configs.keys():
        if k in model_path:
            return sg_x_i_noise_configs[k]

    raise RuntimeError(f'Model {model_path} is not supported with --use_predefined_hp set to True')


def get_ig_baseline(model_path):
    for k in ig_baseline_configs.keys():
        if k in model_path:
            return ig_baseline_configs[k]

    raise RuntimeError(f'Model {model_path} is not supported with --use_predefined_hp set to True')


def get_ks_baseline(model_path):
    for k in ks_baseline_configs.keys():
        if k in model_path:
            return ks_baseline_configs[k]

    raise RuntimeError(f'Model {model_path} is not supported with --use_predefined_hp set to True')


def create_dirs():
    try:
        os.mkdir(args['output_dir'])
        os.mkdir(os.path.join(args['output_dir'], CERTAIN_DIR))
        os.mkdir(os.path.join(args['output_dir'], UNSURE_DIR))
    except OSError:
        print(f'A directory already exists, proceeding')


def prepare_noise_test():
    method_file_dict.clear()
    method_file_dict['sg_50_0.05'] = 'sg_50_0.05_attrs.json'
    method_file_dict['sg_50_0.15'] = 'sg_50_0.15_attrs.json'
    method_file_dict['sg_50_0.25'] = 'sg_50_0.25_attrs.json'
    method_file_dict['sg_50_0.05_x_inputs'] = 'sg_50_0.05_x_inputs_attrs.json'
    method_file_dict['sg_50_0.15_x_inputs'] = 'sg_50_0.15_x_inputs_attrs.json'
    method_file_dict['sg_50_0.25_x_inputs'] = 'sg_50_0.25_x_inputs_attrs.json'


def prepare_ig_baseline_test():
    method_file_dict.clear()
    method_file_dict['ig_50_zero'] = 'ig_50_zero_attrs.json'
    method_file_dict['ig_50_pad'] = 'ig_50_pad_attrs.json'
    method_file_dict['ig_50_avg'] = 'ig_50_avg_attrs.json'
    method_file_dict['ig_50_custom'] = 'ig_50_custom_attrs.json'


def prepare_ks_baseline_test():
    method_file_dict.clear()
    method_file_dict['ks_200_pad'] = 'ks_200_pad_attrs.json'
    method_file_dict['ks_200_mask'] = 'ks_200_mask_attrs.json'
    method_file_dict['ks_200_unk'] = 'ks_200_unk_attrs.json'

#   -----------------------------------------------------------------------------------------------


def get_sentences_tokens_and_phrase_sentiments():
    """
    Loads the needed SST dataset features
    :return:
    """
    with open(os.path.join(args['dataset_dir'], 'sentences_tokens_test.json'), 'r', encoding='utf-8') as f:
        sentences_tokens = json.loads(f.read())

    with open(os.path.join(args['dataset_dir'], 'phrase_sentiments.json'), 'r', encoding='utf-8') as f:
        phrase_sentiments = json.loads(f.read())

    sentences = []
    tokens = []
    for s, t in sentences_tokens:
        sentences.append(s)
        tokens.append(t)

    return sentences, tokens, phrase_sentiments


def get_sentence_sentiments(tokens, phrase_sentiments):
    """
    Given sentence tokens, returns list of sentiment values
    :param tokens:
    :param phrase_sentiments:
    :return:
    """
    output = []
    for token in tokens:
        output.append(phrase_sentiments[token])
    return output


def format_attrs(attrs, sentence):
    """
    Preprocesses the attributions shape and removes cls and sep tokens
    :param attrs:
    :param sentence:
    :return:
    """
    tokenized = tokenizer(sentence)

    if len(attrs.shape) == 2 and attrs.shape[0] == 1:
        attrs = torch.squeeze(attrs)

    attrs_list = attrs.tolist()
    return attrs_list[1:len(tokenized.data['input_ids']) - 1]  # leave out cls and sep


def prepare_embeds_and_att_mask(sentence):
    """
    Prepares attention mask and inputs embeds
    :param sentence:
    :return:
    """
    encoded = tokenizer(sentence, max_length=512, truncation=True, return_tensors='pt')
    attention_mask = encoded.data['attention_mask'].to(device)
    input_ids = torch.squeeze(encoded.data['input_ids']).to(device)
    input_embeds = torch.unsqueeze(torch.index_select(embeddings, 0, input_ids), 0).requires_grad_(True).to(device)
    input_ids.to('cpu')

    return input_embeds, attention_mask


def prepare_input_ids_and_attention_mask(sentence, add_special_tokens=True):
    """
    Prepares input ids and attention mask
    :param sentence:
    :return:
    """
    encoded = tokenizer(sentence, max_length=512, truncation=True, return_tensors='pt')
    attention_mask = encoded.data['attention_mask'].to(device)
    input_ids = encoded.data['input_ids'].to(device)

    # a special case for kernel shap
    if not add_special_tokens:
        input_ids = torch.squeeze(input_ids)
        input_ids = input_ids[1:-1]
        input_ids = torch.unsqueeze(input_ids, dim=0)

    return input_ids, attention_mask


#   -----------------------------------------------------------------------------------------------

def _do_kernel_shap(sentences, target_indices, hf_model, n_steps, baseline_idx, file, target_dir):
    attrs = []
    cls_tensor = torch.tensor([[cls_token_index]]).to(device)
    sep_tensor = torch.tensor([[sep_token_index]]).to(device)
    for sentence, target_idx in zip(sentences, target_indices):
        input_ids, attention_mask = prepare_input_ids_and_attention_mask(sentence, add_special_tokens=False)
        attr = kernel_shap_attributions(input_ids, attention_mask, target_idx, hf_model, baseline_idx,
                                        cls_tensor, sep_tensor, logit_fn, n_steps)
        attr = torch.squeeze(attr)  # no averaging as the attributions are w.r.t. input ids
        attrs.append(format_attrs(attr, sentence))

    with open(str(os.path.join(args['output_dir'], target_dir, method_file_dict[file])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))


def create_kernel_shap_attributions(sentences, target_indices, hf_model, target_dir=CERTAIN_DIR):
    """
    Generates KernelShap attributions
    :param sentences:
    :param target_indices:
    :param model:
    :param target_dir:
    :return:
    """
    baseline_type = args['ks_baseline'] if not args['use_prepared_hp'] else get_ks_baseline(args['model_path'])
    if baseline_type == 'pad':
        baseline = pad_token_index
    elif baseline_type == 'unk':
        baseline = unk_token_index
    elif baseline_type == 'mask':
        baseline = mask_token_index
    else:
        raise RuntimeError(f'Unknown KS baseline {baseline_type}')

    _do_kernel_shap(sentences, target_indices, hf_model, 100, baseline, 'ks_100', target_dir)
    _do_kernel_shap(sentences, target_indices, hf_model, 200, baseline, 'ks_200', target_dir)
    _do_kernel_shap(sentences, target_indices, hf_model, 500, baseline, 'ks_500', target_dir)


def create_kernel_shap_baseline_test_attributions(sentences, target_indices, hf_model, target_dir=CERTAIN_DIR):
    _do_kernel_shap(sentences, target_indices, hf_model, 200, pad_token_index, 'ks_200_pad', target_dir)
    _do_kernel_shap(sentences, target_indices, hf_model, 200, unk_token_index, 'ks_200_unk', target_dir)
    _do_kernel_shap(sentences, target_indices, hf_model, 200, mask_token_index, 'ks_200_mask', target_dir)


def create_gradient_attributions(sentences, target_indices, target_dir=CERTAIN_DIR):
    """
    Generates gradient and gradient x input attributions
    :param sentences:
    :param target_indices:
    :param target_dir:
    :return:
    """
    attrs = []
    for sentence, target_idx in zip(sentences, target_indices):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attr = gradient_attributions(input_embeds, attention_mask, target_idx, model, logit_fn)
        attr = torch.squeeze(attr)
        attr = attr.mean(dim=1)         # average over the embedding attributions
        attrs.append(format_attrs(attr, sentence))

    with open(str(os.path.join(args['output_dir'], target_dir, method_file_dict['grads'])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))

    attrs = []
    for sentence, target_idx in zip(sentences, target_indices):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attr = gradient_attributions(input_embeds, attention_mask, target_idx, model, logit_fn, True)
        attr = torch.squeeze(attr)
        attr = attr.mean(dim=1)         # average over the embedding attributions
        attrs.append(format_attrs(attr, sentence))

    with open(str(os.path.join(args['output_dir'], target_dir, method_file_dict['grads_x_inputs'])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))


def _do_ig(sentences, target_indices, steps, file, target_dir, baseline_type=None):
    """
    Generates integrated gradients attributions given the configuration
    """
    average_emb = embeddings.mean(dim=0)

    attrs = []
    deltas = []
    percs = []
    for sentence, target_idx in zip(sentences, target_indices):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        if baseline_type == 'avg':
            baseline = utils.baselines.embedding_space_average_baseline(input_embeds, average_emb)
        elif baseline_type == 'zero':
            baseline = utils.baselines.zero_embedding_baseline(input_embeds)
        elif baseline_type == 'pad':
            baseline = utils.baselines.pad_baseline(input_embeds, embeddings[103])#embeddings[pad_token_index])
        elif baseline_type == 'custom':
            baseline = utils.baselines.prepared_baseline(input_embeds, args['baselines_dir']).to(device)
        else:
            raise RuntimeError(f'Unknown baseline type: {baseline_type}')

        attr = ig_attributions(input_embeds, attention_mask, target_idx, baseline, model, logit_fn, steps)
        attr = torch.squeeze(attr)

        # temp
        attr_sum = torch.sum(attr)
        score_for_baseline = logit_fn(model(inputs_embeds=baseline, attention_mask=attention_mask).logits)
        score_for_target = logit_fn(model(inputs_embeds=input_embeds, attention_mask=attention_mask).logits)
        diff_target = score_for_target[0][target_idx] - score_for_baseline[0][target_idx]

        if abs(attr_sum / diff_target) == math.inf or abs(attr_sum - diff_target) == math.inf:
            continue

        percs.append(float(attr_sum / diff_target))
        deltas.append(float(attr_sum - diff_target))
        # temp

        attr = attr.mean(dim=1)         # average over the embedding attributions
        attrs.append(format_attrs(attr, sentence))

    print(f'Method: {file} --- Avg perc.: {sum(percs) / len(percs)} --- Avg. delta: {sum(deltas) / len(deltas)} --- Perc std: {torch.std(torch.tensor(percs), dim=0).item()} --- Delta std: {torch.std(torch.tensor(deltas), dim=0).item()}')
    print(f'{torch.std(torch.tensor(percs), dim=0).item()}')
    with open(str(os.path.join(args['output_dir'], target_dir, method_file_dict[file])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))


def create_ig_attributions(sentences, target_indices, target_dir=CERTAIN_DIR):
    """
    Generates all the integrated gradient attributions
    :param sentences:
    :param target_indices:
    :param target_dir:
    :return:
    """
    baseline_type = args['ig_baseline'] if not args['use_prepared_hp'] else get_ig_baseline(args['model_path'])

    _do_ig(sentences, target_indices, 20, 'ig_20', target_dir, baseline_type)
    _do_ig(sentences, target_indices, 50, 'ig_50', target_dir, baseline_type)
    _do_ig(sentences, target_indices, 100, 'ig_100', target_dir, baseline_type)


def create_ig_baseline_test_attributions(sentences, target_indices, target_dir=CERTAIN_DIR):
    _do_ig(sentences, target_indices, 50, 'ig_50_zero', target_dir, baseline_type='zero')
    _do_ig(sentences, target_indices, 50, 'ig_50_pad', target_dir, baseline_type='pad')
    _do_ig(sentences, target_indices, 50, 'ig_50_avg', target_dir, baseline_type='avg')
    _do_ig(sentences, target_indices, 50, 'ig_50_custom', target_dir, baseline_type='custom')


def _do_sg(sentences, target_indices, samples, file, target_dir, noise_level=None, noise_level_x_i=None):
    """
    Generates smoothgrad attributions given the configuration
    :param sentences:
    :param target_indices:
    :param samples:
    :param file:
    :param target_dir:
    :return:
    """
    # If SG and SGxI noise levels are identical, we can calculate the gradients once
    single_pass = noise_level == noise_level_x_i

    attrs = []
    attrs_x_inputs = []
    for sentence, target_idx in zip(sentences, target_indices):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attr = sg_attributions(input_embeds, attention_mask, target_idx, model, logit_fn, samples, noise_level)

        if single_pass:
            attr_x_input = attr.to(device) * input_embeds
            attr_x_input = torch.squeeze(attr_x_input)
            attr_x_input = attr_x_input.mean(dim=1)
            attrs_x_inputs.append(format_attrs(attr_x_input, sentence))

        attr = torch.squeeze(attr)
        attr = attr.mean(dim=1)         # average over the embedding attributions
        attrs.append(format_attrs(attr, sentence))

    # If we need a separate pass for SGxI
    if not single_pass:
        for sentence, target_idx in zip(sentences, target_indices):
            input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
            attr = sg_attributions(input_embeds, attention_mask, target_idx, model, logit_fn, samples, noise_level_x_i)
            attr_x_input = attr.to(device) * input_embeds
            attr_x_input = torch.squeeze(attr_x_input)
            attr_x_input = attr_x_input.mean(dim=1)  # average over the embedding attributions
            attrs_x_inputs.append(format_attrs(attr_x_input, sentence))

    with open(str(os.path.join(args['output_dir'], target_dir, method_file_dict[file])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))

    with open(str(os.path.join(args['output_dir'], target_dir, method_file_dict[file + '_x_inputs'])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs_x_inputs))


def create_smoothgrad_attributions(sentences, target_indices, target_dir=CERTAIN_DIR):
    """
    Generates all the smoothgrad attributions
    :param sentences:
    :param target_indices:
    :param target_dir:
    :return:
    """
    sg_noise = args['sg_noise'] if not args['use_prepared_hp'] else get_sg_noise(args['model_path'])
    sg_x_i_noise = args['sg_noise'] if not args['use_prepared_hp'] else get_sg_x_i_noise(args['model_path'])

    _do_sg(sentences, target_indices, 20, 'sg_20', target_dir, sg_noise, sg_x_i_noise)
    _do_sg(sentences, target_indices, 50, 'sg_50', target_dir, sg_noise, sg_x_i_noise)
    _do_sg(sentences, target_indices, 100, 'sg_100', target_dir, sg_noise, sg_x_i_noise)


def create_smoothgrad_noise_test_attributions(sentences, target_indices, target_dir=CERTAIN_DIR):
    _do_sg(sentences, target_indices, 50, 'sg_50_0.05', target_dir, 0.05)
    _do_sg(sentences, target_indices, 50, 'sg_50_0.15', target_dir, 0.15)
    _do_sg(sentences, target_indices, 50, 'sg_50_0.25', target_dir, 0.25)


def create_relprop_attributions(sentences, target_indices, target_dir=CERTAIN_DIR):
    """
    Generates Chefer et al. attributions
    :param sentences:
    :param target_indices:
    :param target_dir:
    :return:
    """
    attrs = []
    for sentence, target_idx in zip(sentences, target_indices):
        input_ids, attention_mask = prepare_input_ids_and_attention_mask(sentence)
        res = relprop_explainer.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index=target_idx)
        attrs.append(format_attrs(res, sentence))   # no averaging as the attributions are w.r.t. input ids

    with open(str(os.path.join(args['output_dir'], target_dir, method_file_dict['relprop'])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))


#   -----------------------------------------------------------------------------------------------


def main(model):
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

    for sentence, tokens in zip(sentences[:150] + sentences[:-150], tokens[:150] + tokens[:-150]):
        # first classify the sample
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        res = torch.nn.functional.softmax(model(inputs_embeds=input_embeds, attention_mask=attention_mask).logits, dim=-1)
        top_idx = int(torch.argmax(res, dim=-1))
        # compare it to the label
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
    with open(os.path.join(args['output_dir'], CERTAIN_DIR, 'sst_bert_tokens.json'), 'w+', encoding='utf-8') as f:
        f.write(json.dumps({'bert_tokens': bert_tokens_correct, 'sst_tokens': sst_tokens_correct}))
    with open(os.path.join(args['output_dir'], UNSURE_DIR, 'sst_bert_tokens.json'), 'w+', encoding='utf-8') as f:
        f.write(json.dumps({'bert_tokens': bert_tokens_unsure, 'sst_tokens': sst_tokens_unsure}))

    with open(os.path.join(args['output_dir'], 'method_file_dict.json'), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(method_file_dict))

    if args['smoothgrad_noise_test']:
        # special case for testing noise effect on attributions
        create_smoothgrad_noise_test_attributions(correct_pred_sentences, correct_pred_indices)
    elif args['ig_baseline_test']:
        create_ig_baseline_test_attributions(correct_pred_sentences, correct_pred_indices)
    elif args['ks_baseline_test']:
        # we need to switch models for captum
        if isinstance(model, BertForSequenceClassificationChefer):
            del model
            hf_model = AutoModelForSequenceClassification.from_pretrained(args['model_path']).to(device)
            create_kernel_shap_baseline_test_attributions(correct_pred_sentences, correct_pred_indices, hf_model)
        else:
            create_kernel_shap_baseline_test_attributions(correct_pred_sentences, correct_pred_indices, model)
    else:
        # create attributions for the correctly predicted and certain
        #create_gradient_attributions(correct_pred_sentences, correct_pred_indices)
        #create_smoothgrad_attributions(correct_pred_sentences, correct_pred_indices)
        create_ig_attributions(correct_pred_sentences, correct_pred_indices)

        # create attributions for the correctly predicted but uncertain
        #create_gradient_attributions(unsure_pred_sentences, unsure_pred_indices, UNSURE_DIR)
        #create_smoothgrad_attributions(unsure_pred_sentences, unsure_pred_indices, UNSURE_DIR)
        #create_ig_attributions(unsure_pred_sentences, unsure_pred_indices, UNSURE_DIR)

        #if is_relprop_possible(model):
        #    create_relprop_attributions(correct_pred_sentences, correct_pred_indices)
        #    create_relprop_attributions(unsure_pred_sentences, unsure_pred_indices, UNSURE_DIR)
        #else:
        #    method_file_dict.pop('relprop')

        # we need a different model for Captum, potentially - the Chefer implementation registers
        # gradient hooks, and captum uses torch.no_grad(), so we check for Chefer impl. and if needed
        # initialize a new model
        #if isinstance(model, BertForSequenceClassificationChefer):
        #    del model
        #    hf_model = AutoModelForSequenceClassification.from_pretrained(args['model_path']).to(device)
        #    create_kernel_shap_attributions(correct_pred_sentences, correct_pred_indices, hf_model)
        #    create_kernel_shap_attributions(unsure_pred_sentences, unsure_pred_indices, hf_model, UNSURE_DIR)
        #else:
        #    create_kernel_shap_attributions(correct_pred_sentences, correct_pred_indices, model)
        #    create_kernel_shap_attributions(unsure_pred_sentences, unsure_pred_indices, model, UNSURE_DIR)



def parse_bool(s):
    return s.lower() == 'true'


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output_sst_attrs', help='Attributions output directory')
    parser.add_argument('--model_path', required=True, help='Trained model - loaded with from_pretrained')
    parser.add_argument('--baselines_dir', required=True, help='Directory with baseline examples')
    parser.add_argument('--sg_noise', required=False, type=float, default=0.15, help='The noise level applied to smoothgrad')
    parser.add_argument('--ig_baseline', required=False, default='zero', type=str, help='Integrated Gradients baseline - one of [zero, avg, pad, custom]')
    parser.add_argument('--ks_baseline', required=False, default='pad', type=str, help='KernelShap baseline token - one of [pad, unk, mask]')
    parser.add_argument('--use_prepared_hp', required=False, default=False, type=parse_bool, help='Use predetermined hyperparameters')
    parser.add_argument('--dataset_dir', required=False, type=str, default='datasets_ours/sst', help='The default corresponds to the project root')
    parser.add_argument('--smoothgrad_noise_test', required=False, default=False, type=parse_bool, help='Perform smoothgrad noise level test')
    parser.add_argument('--ig_baseline_test', required=False, default=False, type=parse_bool, help='Perform Integrated Gradients baseline test')
    parser.add_argument('--ks_baseline_test', required=False, default=False, type=parse_bool, help='Perform KernelShap baseline test')
    args = vars(parser.parse_args())

    print(args['model_path'])

    # prepare models
    tokenizer = AutoTokenizer.from_pretrained(args['model_path'], local_files_only=True)
    config = AutoConfig.from_pretrained(args['model_path'])
    if 'BertForSequenceClassification' in config.architectures[0]:
        # check if we can apply relprop
        model = BertForSequenceClassificationChefer.from_pretrained(args['model_path'], local_files_only=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args['model_path'], local_files_only=True)
    model = model.to(device)
    model.eval()
    embeddings = model.bert.base_model.embeddings.word_embeddings.weight.data

    # we expect the models to use these tokens
    pad_token_index = tokenizer.pad_token_id
    cls_token_index = tokenizer.cls_token_id
    sep_token_index = tokenizer.sep_token_id
    unk_token_index = tokenizer.unk_token_id
    mask_token_index = tokenizer.mask_token_id

    relprop_explainer = Generator(model)

    logit_fn = torch.nn.Softmax(dim=-1)

    # finish setup and start generating
    create_dirs()
    if args['smoothgrad_noise_test']:
        prepare_noise_test()
    elif args['ig_baseline_test']:
        prepare_ig_baseline_test()
    elif args['ks_baseline_test']:
        prepare_ks_baseline_test()

    main(model)
