import torch
from models.bert_512 import BertSequenceClassifierSST
import os
import argparse


def main(args: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = BertSequenceClassifierSST.from_pretrained(args['model_folder'], num_labels=2, local_files_only=True)
    model = model.to(device)
    model.eval()

    # extract embeddings
    embeddings = model.bert.base_model.embeddings.word_embeddings.weight.data
    embedding_dimensions = embeddings.shape[1]

    os.makedirs(args['output_dir'], exist_ok=True)

    for length in range(args['start'], 129):
        # generate attention mask for the length
        arr = [1 for _ in range(length)]
        attention_mask = torch.tensor([arr]).to(device)

        # generate random embeddings
        baseline = torch.randn((1, length, embedding_dimensions), dtype=torch.float32).to(device).requires_grad_(True)

        # use gradients to modify the input embeddings
        lr = args['lr']
        output = model(baseline, attention_mask=attention_mask)[:, 0]
        res = float(output)
        while abs(res - 0.5) > args['tolerance']:
            grads = torch.autograd.grad(output, baseline)[0]
            # because of softmax and a binary classification problem, we can balance
            # the outputs using gradients like this
            if res < 0.5:
                baseline = lr * grads + baseline
            else:
                baseline = -1 * lr * grads + baseline

            # softmax is applied in the model
            output = model(baseline, attention_mask=attention_mask)[:, 0]
            res = float(output)

        # save
        torch.save(baseline, args['output_dir'] + '/' + f'{length}.pt')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_folder', type=str, required=True, help='Has to be a BertForSequenceClassification')
    argparser.add_argument('--output_dir', type=str, required=True)
    argparser.add_argument('--start', type=int, default=1, required=False, help='Starting length')
    argparser.add_argument('--lr', type=float, default=0.5, required=False, help='Learning rate')
    argparser.add_argument('--tolerance', type=float, default=0.025, required=False, help='How far off 0.5 is permitted')
    args = vars(argparser.parse_args())
    main(args)


