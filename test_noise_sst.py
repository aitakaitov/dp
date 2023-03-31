import argparse
import random

import transformers
import torch

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


def load_sst(tokenizer, embeddings):
    with open('datasets_ours/sst/test.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        random.shuffle(lines)

    inputs_embeds = []
    att_masks = []
    labels = []
    for line in lines:
        text, label = line.strip().split('\t')
        encoded = tokenizer(text, return_tensors='pt')
        embedded = torch.index_select(embeddings, 0, torch.squeeze(encoded['input_ids']).to(device))
        inputs_embeds.append(torch.unsqueeze(embedded, dim=0).to(device))
        att_masks.append(encoded['attention_mask'].to(device))
        labels.append(int(label))

    return inputs_embeds, att_masks, labels


def main():
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args['model_name']).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model_name'])
    embeddings = model.bert.embeddings.word_embeddings.weight.to(device)

    inputs_embeds, att_masks, labels = load_sst(tokenizer, embeddings)
    stdevs = []
    l2s = []
    for input_embed, att_mask, label in zip(inputs_embeds[:100], att_masks[:100], labels[:100]):
        grads_total = None
        output = torch.softmax(model(inputs_embeds=input_embed, attention_mask=att_mask).logits, dim=1)[:, label]
        if output.item() < 0.6:
            continue

        for sample in generate_samples(input_embed, stdev_spread=args['stdev_spread'], samples=args['samples']):
            sample = sample.requires_grad_(True).to(device)
            model.zero_grad()
            output = torch.softmax(model(inputs_embeds=sample, attention_mask=att_mask).logits, dim=1)[:, label]
            grads = torch.autograd.grad(output, sample)[0]

            if grads_total is None:
                grads_total = grads
            else:
                grads_total = torch.concat([grads_total, grads], dim=0)

        # norm over samples
        #l2 = grads_total.norm(p=2, dim=0)
        #grads_total /= l2
        # std over samples
        std = torch.std(grads_total, dim=0)
        # average over emb and seq
        std = torch.mean(std, dim=1)
        std = torch.mean(std, dim=0)
        stdevs.append(std.item())
        #l2s.append(torch.mean(torch.mean(l2, dim=1), dim=0))

    print(f'stdev: {sum(stdevs) / len(stdevs)}')
    #print(f'l2: {sum(l2s) / len(l2s)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name')
    parser.add_argument('--samples', default=100, type=int)
    parser.add_argument('--stdev_spread', default=0.01, type=float)
    parser.add_argument('--k', required=False)
    args = vars(parser.parse_args())

    args['top_k'] = 'k' in args.keys()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args['model_name'] = 'bert-medium-sst-0'
    main()

    args['model_name'] = 'bert-mini-sst-0'
    main()

