import argparse

import torch
import json

import transformers
from transformers import AutoTokenizer, AutoConfig
from attribution_methods_custom import gradient_attributions, ig_attributions, sg_attributions, kernel_shap_attributions
from models.bert_512 import BertForSequenceClassificationChefer
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from utils.check_relprop import is_relprop_possible
from utils.list_utils import count_rec
import utils.baselines

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CERTAIN_DIR = 'certain'
UNSURE_DIR = 'unsure'

UNSURE_PREDICTION_DELTA = 0.1

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
    'Czert': 0.05,
    'MiniLM': 0.05,
    'small-e-czech': 0.05
}


sg_x_i_noise_configs = {
    'Czert': 0.05,
    'MiniLM': 0.15,
    'small-e-czech': 0.15
}


ig_baseline_configs = {
    'Czert': 'zero',
    'MiniLM': 'pad',
    'small-e-czech': 'pad'
}

ks_baseline_configs = {
    'Czert': 'mask',
    'MiniLM': 'mask',
    'small-e-czech': 'pad'
}


def get_sg_x_i_noise(model_path):
    for k in sg_x_i_noise_configs.keys():
        if k in model_path:
            return sg_x_i_noise_configs[k]

    raise RuntimeError(f'Model {model_path} is not supported with --use_predefined_hp set to True')


def get_sg_noise(model_path):
    for k in sg_noise_configs.keys():
        if k in model_path:
            return sg_noise_configs[k]

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
    method_file_dict['ks_200_unk'] = 'ks_200_unk_attrs.json'
    method_file_dict['ks_200_mask'] = 'ks_200_mask_attrs.json'


#   -----------------------------------------------------------------------------------------------


def parse_csv_line(line: str):
    split = line.strip('\n').split('~')
    text = split[0]
    classes = split[1:]
    return text, classes


def get_data():
    with open(os.path.join(args['dataset_dir'], 'classes.json'), 'r', encoding='utf-8') as f:
        class_dict = json.loads(f.read())

    with open(os.path.join(args['dataset_dir'], 'test.csv'), 'r', encoding='utf-8') as f:
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
    """
    Prepares input embeddings and attention mask
    :param sentence:
    :return:
    """
    encoded = tokenizer(sentence, max_length=512, truncation=True, return_tensors='pt')
    attention_mask = encoded.data['attention_mask'].to(device)
    input_embeds = torch.unsqueeze(torch.index_select(embeddings, 0, torch.squeeze(encoded.data['input_ids']).to(device)), 0).requires_grad_(True).to(device)

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

def create_gradient_attributions(sentences, target_indices_list, target_dir=CERTAIN_DIR):
    """
    Creates vanilla gradient attributions
    :param sentences:
    :param target_indices_list:
    :param target_dir:
    :return:
    """
    attrs = []
    for sentence, target_indices in zip(sentences, target_indices_list):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attrs_temp = []
        for target_idx in target_indices:
            attr = gradient_attributions(input_embeds, attention_mask, target_idx, model, logit_fn)
            attr = torch.squeeze(attr)
            attr = attr.mean(dim=1)     # average over embeddings
            attrs_temp.append(format_attrs(attr, sentence))

        attrs.append(attrs_temp)

    with open(os.path.join(args['output_dir'], target_dir, method_file_dict['grads']), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))

    attrs = []
    for sentence, target_indices in zip(sentences, target_indices_list):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attrs_temp = []
        for target_idx in target_indices:
            attr = gradient_attributions(input_embeds, attention_mask, target_idx, model, logit_fn, True)
            attr = torch.squeeze(attr)
            attr = attr.mean(dim=1)     # average over embeddings
            attrs_temp.append(format_attrs(attr, sentence))

        attrs.append(attrs_temp)

    with open(os.path.join(args['output_dir'], target_dir, method_file_dict['grads_x_inputs']), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))


def _do_ig(sentences, target_indices_list, steps, file, target_dir, baseline_type=None):
    average_emb = embeddings.mean(dim=0)
    attrs = []
    for sentence, target_indices in zip(sentences, target_indices_list):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attrs_temp = []
        for target_idx in target_indices:
            if baseline_type == 'avg':
                baseline = utils.baselines.embedding_space_average_baseline(input_embeds, average_emb)
            elif baseline_type == 'zero':
                baseline = utils.baselines.zero_embedding_baseline(input_embeds)
            elif baseline_type == 'pad':
                baseline = utils.baselines.pad_baseline(input_embeds, embeddings[pad_token_index])
            elif baseline_type == 'custom':
                baseline = utils.baselines.prepared_baseline(input_embeds, args['baselines_dir']).to(device)
            else:
                raise RuntimeError(f'Unknown baseline type: {baseline_type}')
            attr = ig_attributions(input_embeds, attention_mask, target_idx, baseline, model, logit_fn, steps)
            attr = torch.squeeze(attr)
            attr = attr.mean(dim=1)     # average over embeddings
            attrs_temp.append(format_attrs(attr, sentence))

        attrs.append(attrs_temp)

    with open(os.path.join(args['output_dir'], target_dir, method_file_dict[file]), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))


def create_ig_attributions(sentences, target_indices, target_dir=CERTAIN_DIR):
    """
    Generates Integrated Gradients attributions
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


def _do_sg(sentences, target_indices_list, samples, file, target_dir, noise_level=None, noise_level_x_i=None):
    single_pass = noise_level == noise_level_x_i

    attrs = []
    attrs_x_inputs = []
    for sentence, target_indices in zip(sentences, target_indices_list):
        inputs_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        temp_attrs = []
        temp_attrs_x_inputs = []
        for target_idx in target_indices:
            attr = sg_attributions(inputs_embeds, attention_mask, target_idx, model, logit_fn, samples, noise_level)

            if single_pass:
                attr_x_input = attr.to(device) * inputs_embeds
                attr_x_input = torch.squeeze(attr_x_input)
                attr_x_input = attr_x_input.mean(dim=1)     # average over embeddings
                temp_attrs_x_inputs.append(format_attrs(attr_x_input, sentence))

            attr = torch.squeeze(attr)
            attr = attr.mean(dim=1)     # average over embeddings
            temp_attrs.append(format_attrs(attr, sentence))

        attrs.append(temp_attrs)

        if single_pass:
            attrs_x_inputs.append(temp_attrs_x_inputs)

    with open(os.path.join(args['output_dir'], target_dir, method_file_dict[file]), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))

    if not single_pass:
        for sentence, target_indices in zip(sentences, target_indices_list):
            inputs_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
            temp_attrs_x_inputs = []
            for target_idx in target_indices:
                attr = sg_attributions(inputs_embeds, attention_mask, target_idx, model, logit_fn, samples, noise_level)
                attr_x_input = attr.to(device) * inputs_embeds
                attr_x_input = torch.squeeze(attr_x_input)
                attr_x_input = attr_x_input.mean(dim=1)     # average over embeddings
                temp_attrs_x_inputs.append(format_attrs(attr_x_input, sentence))

            attrs_x_inputs.append(temp_attrs_x_inputs)

    with open(os.path.join(args['output_dir'], target_dir, method_file_dict[file + '_x_inputs']), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs_x_inputs))


def create_smoothgrad_attributions(sentences, target_indices, target_dir=CERTAIN_DIR):
    """
    Generates SmoothGRAD attributions
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


def _do_kernel_shap(sentences, target_indices_list, model, n_steps, baseline_idx, file, target_dir):
    attrs = []
    cls_tensor = torch.tensor([[cls_token_index]]).to(device)
    sep_tensor = torch.tensor([[sep_token_index]]).to(device)
    for sentence, target_indices in zip(sentences, target_indices_list):
        input_ids, attention_mask = prepare_input_ids_and_attention_mask(sentence, add_special_tokens=False)
        temp_attrs = []
        for target_idx in target_indices:
            attr = kernel_shap_attributions(input_ids, attention_mask, target_idx, model, baseline_idx,
                                            cls_tensor, sep_tensor, torch.nn.Softmax(dim=-1), n_steps)
            attr = torch.squeeze(attr)  # no averaging as the attributions are w.r.t. input ids
            temp_attrs.append(format_attrs(attr, sentence))

        attrs.append(temp_attrs)

    with open(str(os.path.join(args['output_dir'], target_dir, method_file_dict[file])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))


def create_kernel_shap_attributions(sentences, target_indices, model, target_dir=CERTAIN_DIR):
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

    _do_kernel_shap(sentences, target_indices, model, 100, baseline, 'ks_100', target_dir)
    _do_kernel_shap(sentences, target_indices, model, 200, baseline, 'ks_200', target_dir)
    _do_kernel_shap(sentences, target_indices, model, 500, baseline, 'ks_500', target_dir)


def create_kernel_shap_baseline_test_attributions(sentences, target_indices, model, target_dir=CERTAIN_DIR):
    _do_kernel_shap(sentences, target_indices, model, 200, pad_token_index, 'ks_200_pad', target_dir)
    _do_kernel_shap(sentences, target_indices, model, 200, unk_token_index, 'ks_200_unk', target_dir)
    _do_kernel_shap(sentences, target_indices, model, 200, mask_token_index, 'ks_200_mask', target_dir)


def create_relprop_attributions(sentences, target_indices_list, target_dir=CERTAIN_DIR):
    """
    Generates Chefer et al attributions
    :param sentences:
    :param target_indices_list:
    :param target_dir:
    :return:
    """
    attrs = []
    for sentence, target_indices in zip(sentences, target_indices_list):
        input_ids, attention_mask = prepare_input_ids_and_attention_mask(sentence)
        inputs_embeds, _ = prepare_embeds_and_att_mask(sentence)
        temp_attrs = []
        for target_idx in target_indices:
            res = relprop_explainer.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index=target_idx)
            temp_attrs.append(format_attrs(res, sentence))

        attrs.append(temp_attrs)

    with open(os.path.join(args['output_dir'], target_dir, method_file_dict['relprop']), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))

#   -----------------------------------------------------------------------------------------------


def main(model):
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

    for document, label in zip(documents[:20], labels[:20]):
        # check the length - no longer than 512 tokens
        if len(tokenizer.tokenize(document)) + 2 > 512:
            continue
        else:
            labels_short_enough.append(label)

        # first classify the sample
        input_embeds, attention_mask = prepare_embeds_and_att_mask(document)
        res = logit_fn(model(inputs_embeds=input_embeds, attention_mask=attention_mask).logits)
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

    if args['smoothgrad_noise_test']:
        create_smoothgrad_noise_test_attributions(valid_documents_certain, target_indices_certain)
    elif args['ig_baseline_test']:
        create_ig_baseline_test_attributions(valid_documents_certain, target_indices_certain)
    elif args['ks_baseline_test']:
        # Use the HF model for captum
        if isinstance(model, BertForSequenceClassificationChefer):
            del model
            hf_model = transformers.AutoModelForSequenceClassification.from_pretrained(args['model_path']).to(device)
            create_kernel_shap_baseline_test_attributions(valid_documents_certain, target_indices_certain, hf_model)
        else:
            create_kernel_shap_baseline_test_attributions(valid_documents_certain, target_indices_certain, model)
    else:
        #create_gradient_attributions(valid_documents_certain, target_indices_certain)
        #create_smoothgrad_attributions(valid_documents_certain, target_indices_certain)
        #create_ig_attributions(valid_documents_certain, target_indices_certain)

        #create_gradient_attributions(valid_documents_unsure, target_indices_unsure, UNSURE_DIR)
        #create_smoothgrad_attributions(valid_documents_unsure, target_indices_unsure, UNSURE_DIR)
        #create_ig_attributions(valid_documents_unsure, target_indices_unsure, UNSURE_DIR)

        if is_relprop_possible(model):
            create_relprop_attributions(valid_documents_certain, target_indices_certain)
            create_relprop_attributions(valid_documents_unsure, target_indices_unsure, UNSURE_DIR)
        else:
            method_file_dict.pop('relprop')

        # use the HF model for captum
        if isinstance(model, BertForSequenceClassificationChefer):
            del model
            hf_model = transformers.AutoModelForSequenceClassification.from_pretrained(args['model_path']).to(device)
            create_kernel_shap_attributions(valid_documents_certain, target_indices_certain, hf_model)
            create_kernel_shap_attributions(valid_documents_unsure, target_indices_unsure, hf_model, UNSURE_DIR)
        else:
            create_kernel_shap_attributions(valid_documents_certain, target_indices_certain, model)
            create_kernel_shap_attributions(valid_documents_unsure, target_indices_unsure, model, UNSURE_DIR)

    with open(os.path.join(args['output_dir'], 'method_file_dict.json'), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(method_file_dict))


def parse_bool(s):
    return s.lower() == 'true'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output_news_attrs', help='Attributions output directory')
    parser.add_argument('--model_path', required=True, help='Trained model - loaded with from_pretrained')
    parser.add_argument('--sg_noise', required=False, type=float, default=0.15, help='The noise level applied to smoothgrad')
    parser.add_argument('--ig_baseline', required=False, default='zero', type=str, help='Integrated Gradients baseline - one of [zero, avg, pad, custom]')
    parser.add_argument('--ks_baseline', required=False, default='pad', type=str, help='KernelShap baseline token - one of [pad, unk, mask]')
    parser.add_argument('--use_prepared_hp', required=False, default=True, type=parse_bool, help='Use predetermined hyperparameters')
    parser.add_argument('--smoothgrad_noise_test', required=False, default=False, type=parse_bool, help='Perform smoothgrad noise level test')
    parser.add_argument('--ig_baseline_test', required=False, default=False, type=parse_bool, help='Perform Integrated Gradients baseline test')
    parser.add_argument('--ks_baseline_test', required=False, default=False, type=parse_bool, help='Perform KernelShap baseline test')
    parser.add_argument('--baselines_dir', required=True, help='Directory with baseline examples')
    parser.add_argument('--dataset_dir', required=False, default='datasets_ours/news', help='The default corresponds to the project root')

    args = vars(parser.parse_args())

    try:
        os.mkdir(args['output_dir'])
        os.mkdir(os.path.join(args['output_dir'], CERTAIN_DIR))
        os.mkdir(os.path.join(args['output_dir'], UNSURE_DIR))
    except OSError:
        pass

    config = AutoConfig.from_pretrained(args['model_path'])

    if any(['Electra' in arch for arch in config.architectures]):
        tokenizer = AutoTokenizer.from_pretrained(args['model_path'])
        model = transformers.ElectraForSequenceClassification.from_pretrained(args['model_path'])
        embeddings = model.electra.base_model.embeddings.word_embeddings.weight.data
    elif any(['Bert' in arch for arch in config.architectures]):
        tokenizer = AutoTokenizer.from_pretrained(args['model_path'])
        model = BertForSequenceClassificationChefer.from_pretrained(args['model_path'])
        embeddings = model.bert.base_model.embeddings.word_embeddings.weight.data
    elif any(['Roberta' in arch for arch in config.architectures]):
        tokenizer = AutoTokenizer.from_pretrained(args['model_path'])
        model = transformers.XLMRobertaForSequenceClassification.from_pretrained(args['model_path'])
        embeddings = model.roberta.base_model.embeddings.word_embeddings.weight.data
    else:
        raise RuntimeError(f'Architectures {config.architectures} not supported')

    pad_token_index = tokenizer.pad_token_id
    cls_token_index = tokenizer.cls_token_id
    sep_token_index = tokenizer.sep_token_id
    unk_token_index = tokenizer.unk_token_id
    mask_token_index = tokenizer.mask_token_id

    embeddings = embeddings.to(device)
    model = model.to(device)

    model.eval()
    relprop_explainer = Generator(model)

    logit_fn = torch.nn.Sigmoid()

    if args['smoothgrad_noise_test']:
        prepare_noise_test()
    elif args['ig_baseline_test']:
        prepare_ig_baseline_test()
    elif args['ks_baseline_test']:
        prepare_ks_baseline_test()

    main(model)
