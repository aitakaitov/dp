import torch
import transformers
from models.bert_512 import BertSequenceClassifierSST
import os
import argparse


def main(args: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['tokenizer'])
    model = BertSequenceClassifierSST.from_pretrained(args['model_folder'], num_labels=2, local_files_only=True)
    model = model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim=1).to(device)

    # extract padding token
    embeddings = model.bert.base_model.embeddings.word_embeddings.weight.data
    embedding_dimensions = embeddings.shape[1]

    padding_embedding = tokenizer.convert_tokens_to_ids('[PAD]')
    padding_embedding = torch.index_select(embeddings, 0, torch.tensor(padding_embedding).to(device))


    os.makedirs(args['output_dir'], exist_ok=True)

    for length in [279, 380, 155, 505, 425, 149]:
        # generate attention mask for the length
        arr = [1 for _ in range(length)]
        arr.extend([0 for _ in range(512 - length)])
        attention_mask = torch.tensor([arr]).to(device)

        # generate random embeddings and add padding embeddings
        rnd = torch.randn((1, length, embedding_dimensions), dtype=torch.float32).to(device)
        padding = torch.unsqueeze(padding_embedding.repeat((512 - length, 1)), 0).to(device)
        baseline = torch.cat((rnd, padding), 1).to(device).requires_grad_(True)

        # use gradients to modify the input embeddings
        lr = args['lr']
        output = softmax(model(baseline, attention_mask=attention_mask))[:, 0]
        res = float(output)
        while abs(res - 0.5) > args['tolerance']:
            grads = torch.autograd.grad(output, baseline)[0]
            if res < 0.5:
                baseline = lr * grads + baseline
            else:
                baseline = -1 * lr * grads + baseline

            output = softmax(model(baseline, attention_mask=attention_mask))[:, 0]
            res = float(output)
            print(res)

        torch.save(baseline, args['output_dir'] + '/' + f'{length}.pt')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--tokenizer', type=str, required=True)
    argparser.add_argument('--model_folder', type=str, required=True)
    argparser.add_argument('--output_dir', type=str, required=True)
    argparser.add_argument('--start', type=int, default=1, required=False)
    argparser.add_argument('--lr', type=float, default=0.5, required=False)
    argparser.add_argument('--tolerance', type=float, default=0.025, required=False)
    args = vars(argparser.parse_args())
    main(args)


