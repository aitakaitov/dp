import torch
import json
from transformers import AutoTokenizer
from attribution_methods_custom import gradient_attributions, ig_attributions, sg_attributions
from models.bert_512 import BertSequenceClassifierSST
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
import utils.baselines

import os
import argparse

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
    'relprop':  'relprop_attrs.json'
}


def create_dirs():
    os.mkdir(args['output_dir'])
    os.mkdir(os.path.join(args['output_dir'], CERTAIN_DIR))
    os.mkdir(os.path.join(args['output_dir'], UNSURE_DIR))


def prepare_noise_test():
    method_file_dict.clear()
    method_file_dict['sg_50_0.05'] = 'sg_50_0.05_attrs.json'
    method_file_dict['sg_50_0.15'] = 'sg_50_0.15_attrs.json'
    method_file_dict['sg_50_0.25'] = 'sg_50_0.25_attrs.json'
    method_file_dict['sg_50_0.05_x_inputs'] = 'sg_50_0.05_x_inputs_attrs.json'
    method_file_dict['sg_50_0.15_x_inputs'] = 'sg_50_0.15_x_inputs_attrs.json'
    method_file_dict['sg_50_0.25_x_inputs'] = 'sg_50_0.25_x_inputs_attrs.json'

def prepare_baseline_test():
    method_file_dict.clear()
    method_file_dict['ig_50_zero'] = 'ig_50_zero_attrs.json'
    method_file_dict['ig_50_pad'] = 'ig_50_pad_attrs.json'
    method_file_dict['ig_50_avg'] = 'ig_50_avg_attrs.json'
    method_file_dict['ig_50_custom'] = 'ig_50_custom_attrs.json'

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


def prepare_input_ids_and_attention_mask(sentence):
    """
    Prepares input ids and attention mask
    :param sentence:
    :return:
    """
    encoded = tokenizer(sentence, max_length=512, truncation=True, return_tensors='pt')
    attention_mask = encoded.data['attention_mask'].to(device)
    input_ids = encoded.data['input_ids'].to(device)

    return input_ids, attention_mask


def create_neutral_baseline(sentence):
    """
    Loads a precomputed baseline for the given length
    :param sentence:
    :return:
    """
    tokenized = tokenizer(sentence)
    length = len(tokenized.data['input_ids'])

    return torch.load(os.path.join(args['baselines_dir'], str(length) + '.pt')).to(device)


#   -----------------------------------------------------------------------------------------------

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
        attr = gradient_attributions(input_embeds, attention_mask, target_idx, model)
        attr = torch.squeeze(attr)
        attrs.append(format_attrs(attr, sentence))

    with open(str(os.path.join(args['output_dir'], target_dir, method_file_dict['grads'])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))

    attrs = []
    for sentence, target_idx in zip(sentences, target_indices):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attr = gradient_attributions(input_embeds, attention_mask, target_idx, model, True)
        attr = torch.squeeze(attr)
        attrs.append(format_attrs(attr, sentence))

    with open(str(os.path.join(args['output_dir'], target_dir, method_file_dict['grads_x_inputs'])), 'w+', encoding='utf-8') as f:
        f.write(json.dumps(attrs))


def _do_ig(sentences, target_indices, steps, file, target_dir, baseline_type=None):
    """
    Generates integrated gradients attributions given the configuration
    :param sentences:
    :param target_indices:
    :param steps:
    :param file:
    :param target_dir:
    :return:
    """
    average_emb = embeddings.mean(dim=0)

    attrs = []
    for sentence, target_idx in zip(sentences, target_indices):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        if baseline_type is None:
            baseline = create_neutral_baseline(sentence)
        elif baseline_type == 'avg':
            baseline = utils.baselines.embedding_space_average_baseline(input_embeds, average_emb)
        elif baseline_type == 'zero':
            baseline = utils.baselines.zero_embedding_baseline(input_embeds)
        elif baseline_type == 'pad':
            baseline = utils.baselines.pad_baseline(input_embeds, embeddings[pad_token_index])
        elif baseline_type == 'custom':
            baseline = utils.baselines.prepared_baseline(input_embeds, args['baselines_dir']).to(device)
        else:
            raise RuntimeError(f'Unknown baseline type: {baseline_type}')

        attr = ig_attributions(input_embeds, attention_mask, target_idx, baseline, model, steps)
        attr = torch.squeeze(attr)
        attrs.append(format_attrs(attr, sentence))

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
    _do_ig(sentences, target_indices, 20, 'ig_20', target_dir)
    _do_ig(sentences, target_indices, 50, 'ig_50', target_dir)
    _do_ig(sentences, target_indices, 100, 'ig_100', target_dir)


def create_ig_baseline_test_attributions(sentences, target_indices, target_dir=CERTAIN_DIR):
    _do_ig(sentences, target_indices, 50, 'ig_50_zero', target_dir, baseline_type='zero')
    _do_ig(sentences, target_indices, 50, 'ig_50_pad', target_dir, baseline_type='pad')
    _do_ig(sentences, target_indices, 50, 'ig_50_avg', target_dir, baseline_type='avg')
    _do_ig(sentences, target_indices, 50, 'ig_50_custom', target_dir, baseline_type='custom')


def _do_sg(sentences, target_indices, samples, file, target_dir, noise_level=None):
    """
    Generates smoothgrad attributions given the configuration
    :param sentences:
    :param target_indices:
    :param samples:
    :param file:
    :param target_dir:
    :return:
    """
    attrs = []
    attrs_x_inputs = []
    for sentence, target_idx in zip(sentences, target_indices):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attr = sg_attributions(input_embeds, attention_mask, target_idx, model, samples, noise_level)
        attr_x_input = attr.to(device) * input_embeds
        attr_x_input = torch.squeeze(attr_x_input)
        attr = torch.squeeze(attr)
        attrs.append(format_attrs(attr, sentence))
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
    _do_sg(sentences, target_indices, 20, 'sg_20', target_dir, args['sg_noise'])
    _do_sg(sentences, target_indices, 50, 'sg_50', target_dir, args['sg_noise'])
    _do_sg(sentences, target_indices, 100, 'sg_100', target_dir, args['sg_noise'])


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
        inputs_embeds, _ = prepare_embeds_and_att_mask(sentence)
        res = relprop_explainer.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index=target_idx)
        attrs.append(format_attrs(res, sentence))

    with open(str(os.path.join(args['output_dir'], target_dir, method_file_dict['relprop'])), 'w+', encoding='utf-8') as f:
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
    else:
        # create attributions for the correctly predicted and certain
        create_gradient_attributions(correct_pred_sentences, correct_pred_indices)
        create_smoothgrad_attributions(correct_pred_sentences, correct_pred_indices)
        create_ig_attributions(correct_pred_sentences, correct_pred_indices)
        create_relprop_attributions(correct_pred_sentences, correct_pred_indices)

        # create attributions for the correctly predicted but uncertain
        create_gradient_attributions(unsure_pred_sentences, unsure_pred_indices, UNSURE_DIR)
        create_smoothgrad_attributions(unsure_pred_sentences, unsure_pred_indices, UNSURE_DIR)
        create_ig_attributions(unsure_pred_sentences, unsure_pred_indices, UNSURE_DIR)
        create_relprop_attributions(unsure_pred_sentences, unsure_pred_indices, UNSURE_DIR)

    # print document counts
    print(f'Total documents: {len(sentences)}')
    print(f'Correctly predicted documents: {len(correct_pred_sentences) + len(unsure_pred_sentences)}')
    print(f'Documents predicted certainly: {len(correct_pred_sentences)}')
    print(f'Documents predicted unsurely: {len(unsure_pred_sentences)}')


if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output_sst_attrs', help='Attributions output directory')
    parser.add_argument('--model_path', required=True, help='Trained model - loaded with from_pretrained')
    parser.add_argument('--baselines_dir', required=True, help='Directory with baseline examples')
    parser.add_argument('--sg_noise', required=False, type=float, default=0.15)
    parser.add_argument('--dataset_dir', required=False, type=str, default='datasets_ours/sst', help='The default'
                                                                                                     'corresponds to'
                                                                                                     'the project'
                                                                                                     'root')
    parser.add_argument('--smoothgrad_noise_test', required=False, default=False)
    parser.add_argument('--ig_baseline_test', required=False, default=False)
    args = vars(parser.parse_args())

    # prepare models
    tokenizer = AutoTokenizer.from_pretrained(args['model_path'], local_files_only=True)
    model = BertSequenceClassifierSST.from_pretrained(args['model_path'], local_files_only=True)
    model = model.to(device)
    model.eval()
    embeddings = model.bert.base_model.embeddings.word_embeddings.weight.data

    pad_token_index = tokenizer.pad_token_id

    relprop_explainer = Generator(model)

    # finish setup and start generating
    create_dirs()
    if args['smoothgrad_noise_test']:
        prepare_noise_test()
    elif args['ig_baseline_test']:
        prepare_baseline_test()
    main()
