import torch
import transformers
import os
import argparse
import json


def main(args: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('datasets_ours/news/classes.json', 'r', encoding='utf-8') as f:
        class_dict = json.loads(f.read())

    class_count = len(class_dict.keys())

    # load tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['tokenizer'])
    config = transformers.AutoConfig.from_pretrained(args['model_folder'])
    if any('Bert' in arch for arch in config.architectures):
        model = transformers.BertForSequenceClassification.from_pretrained(args['model_folder'], num_labels=class_count,
                                                                            local_files_only=True).to(device)
        embeddings = model.bert.base_model.embeddings.word_embeddings.weight.data

    elif any('Electra' in arch for arch in config.architectures):
        model = transformers.ElectraForSequenceClassification.from_pretrained(args['model_folder'], num_labels=class_count,
                                                                local_files_only=True).to(device)
        embeddings = model.electra.base_model.embeddings.word_embeddings.weight.data

    elif any('Roberta' in arch for arch in config.architectures):
        model = transformers.XLMRobertaForSequenceClassification.from_pretrained(args['model_folder'], num_labels=class_count,
                                                              local_files_only=True).to(device)
        embeddings = model.roberta.base_model.embeddings.word_embeddings.weight.data

    else:
        raise RuntimeError('Architecture not supported')

    model.eval()

    # extract padding token
    embedding_dimensions = embeddings.shape[1]

    padding_embedding = tokenizer.convert_tokens_to_ids('[PAD]')
    padding_embedding = torch.index_select(embeddings, 0, torch.tensor(padding_embedding).to(device))

    os.makedirs(args['output_dir'], exist_ok=True)

    target_label = torch.zeros(1, class_count).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    sigmoid = torch.nn.Sigmoid().to(device)

    for length in range(args['start'], 513):
        # generate attention mask for the length
        arr = [1 for _ in range(length)]
        arr.extend([0 for _ in range(512 - length)])
        attention_mask = torch.tensor([arr]).to(device)

        # generate random embeddings and add padding embeddings
        baseline = torch.randn((1, length, embedding_dimensions), dtype=torch.float32).to(device)

        optimizer = torch.optim.Adam([baseline], lr=args['lr'])

        while True:
            output = sigmoid(model(inputs_embeds=baseline, attention_mask=attention_mask).logits)
            loss = loss_fn(output, target_label)

            if float(torch.max(output)) < args['max']:
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(baseline, args['output_dir'] + '/' + f'{length}.pt')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--tokenizer', type=str, required=True)
    argparser.add_argument('--model_folder', type=str, required=True)
    argparser.add_argument('--output_dir', type=str, required=True)
    argparser.add_argument('--start', type=int, default=1, required=False)
    argparser.add_argument('--lr', type=float, default=1e-1, required=False)
    argparser.add_argument('--max', type=float, default=1e-2, required=False)

    args = vars(argparser.parse_args())
    main(args)


