import argparse
import json
import random

import transformers
import torch
from datasets_ours.news.news_dataset import NewsDataset
from attribution_methods_custom import gradient_attributions

torch.manual_seed(42)
random.seed(42)


def generate_samples(inputs_embeds, samples, stdev_spread):
    result = []
    cpu_embeds = inputs_embeds.to('cpu')
    stdev = stdev_spread * (torch.max(inputs_embeds) - torch.min(inputs_embeds))
    means = torch.zeros((1, inputs_embeds.shape[0], inputs_embeds.shape[2])).to('cpu')
    stdevs = torch.full((1, inputs_embeds.shape[0], inputs_embeds.shape[2]), float(stdev)).to('cpu')
    for i in range(samples):
        noise = torch.normal(means, stdevs).to('cpu')
        result.append(cpu_embeds + noise)

    return result


def load_ctdc(tokenizer):
    with open('datasets_ours/news/test.csv', 'r', encoding='utf-8') as f:
        lines = f.read()

    with open('datasets_ours/news/classes.json', 'r', encoding='utf-8') as f:
        clss = json.loads(f.read())

    dataset = NewsDataset(lines, tokenizer, clss)

    return dataset


def main():
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args['model_name']).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model_name'])
    embeddings = model.bert.embeddings.word_embeddings.weight.to(device)

    dataset = load_ctdc(tokenizer)
    stdevs = []
    l2s = []
    for input_embed, att_mask, label in zip(inputs_embeds[:100], att_masks[:100], labels[:100]):
        grads_total = None
        output = torch.softmax(model(inputs_embeds=input_embed, attention_mask=att_mask).logits, dim=1)[:, label]
        if output.item() < 0.6:
            continue

        if args['top_k']:
            pass

        for sample in generate_samples(input_embed, stdev_spread=args['stdev_spread'], samples=args['samples']):
            sample = sample.requires_grad_(True).to(device)
            model.zero_grad()
            output = torch.softmax(model(inputs_embeds=sample, attention_mask=att_mask).logits, dim=1)[:, label]
            grads = torch.autograd.grad(output, sample)[0]
            grads = torch.sum(grads, dim=2)
            l2 = grads.norm(p=2, dim=1)
            grads = grads / l2
            l2s.append(l2.item())

            if grads_total is None:
                grads_total = grads
            else:
                grads_total = torch.concat([grads_total, grads], dim=0)

        # std over samples
        std = torch.std(grads_total, dim=0)
        # average over seq len
        std = torch.mean(std, dim=0)
        stdevs.append(std.item())

    print(f'stdev: {sum(stdevs) / len(stdevs)}')
    print(f'l2: {sum(l2s) / len(l2s)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name')
    parser.add_argument('--samples', default=100, type=int)
    parser.add_argument('--stdev_spread', default=0.01, type=float)
    parser.add_argument('--k', required=False)
    args = vars(parser.parse_args())

    args['top_k'] = 'k' in args.keys()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args['model_name'] = 'bert-base-cased-sst-0'
    main()

    args['model_name'] = 'bert-mini-sst-0'
    main()
