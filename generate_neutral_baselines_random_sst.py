import os

import torch
import transformers
from models.bert_512 import BertSequenceClassifierSST
import argparse


def main(args: dict):
    print(f'Model folder: {args["model_folder"]}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # prepare
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['tokenizer'])
    model = BertSequenceClassifierSST.from_pretrained(args['model_folder'], num_labels=2, local_files_only=True)
    model = model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim=1).to(device)

    # extract embeddings and get dimensions
    embeddings = model.bert.base_model.embeddings.word_embeddings.weight.data
    embedding_dimensions = embeddings.shape[1]

    # get pad token id and the pad embedding
    padding_embedding = tokenizer.convert_tokens_to_ids('[PAD]')
    padding_embedding = torch.index_select(embeddings, 0, torch.tensor(padding_embedding).to(device))

    # prepare output directory
    OUTPUT_DIR = args['output_dir']
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for length in range(args['start'], 513):
        # create an attention mask for a given length
        arr = [1 for _ in range(length)]
        arr.extend([0 for _ in range(512 - length)])
        attention_mask = torch.tensor([arr]).to(device)

        # generate random sequence
        rnd = torch.rand((1, length, embedding_dimensions), dtype=torch.float32).to('cpu')
        padding = torch.unsqueeze(padding_embedding.repeat((512 - length, 1)), 0).to('cpu')
        baseline = torch.cat((rnd, padding), 1).to(device).requires_grad_(True)

        # get a prediction
        output = softmax(model(baseline, attention_mask=attention_mask))[:, 0]
        res = float(output)

        # while the prediction is 'tolerance-far' from neutral (0.5), generate new random baselines
        while abs(res - 0.5) > args['tolerance']:
            rnd = torch.randn((1, length, embedding_dimensions), dtype=torch.float32).to('cpu')
            padding = torch.unsqueeze(padding_embedding.repeat((512 - length, 1)), 0).to('cpu')
            baseline = torch.cat((rnd, padding), 1).to(device).requires_grad_(True)

            output = softmax(model(baseline, attention_mask=attention_mask))[:, 0]
            res = float(output)

        # save the neutral baseline tensor
        torch.save(baseline, OUTPUT_DIR + '/' + f'{length}.pt')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--tokenizer', required=True, type=str)
    argparser.add_argument('--model_folder', required=True, type=str)
    argparser.add_argument('--output_dir', required=True, type=str)
    argparser.add_argument('--tolerance', required=False, default=0.025, type=float)
    argparser.add_argument('--start', required=False, default=1, type=int)
    args = vars(argparser.parse_args())
    main(args)





