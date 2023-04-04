import json
import argparse
import torchmetrics
import transformers
import torch
from datasets_ours.news.news_dataset import NewsDataset
import wandb

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')

# Disable all GPUS - tensorflow tends to reserve all the GPU memory
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for tf_device in visible_devices:
    assert tf_device.device_type != 'GPU'


def get_class_dict():
    with open('datasets_ours/news/classes.json', 'r', encoding='utf-8') as f:
        class_dict = json.loads(f.read())
    return class_dict


def get_file_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def main():
    test_set = NewsDataset(get_file_text('datasets_ours/news/test.csv'), tokenizer, classes_dict)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_metric = torchmetrics.F1Score().to(device)

    for i in range(len(test_set)):
        label = test_set.labels[i].to(device)
        input_ids = test_set.texts[i][0].to(device)
        attention_mask = test_set.texts[i][1].to(device)

        output = torch.sigmoid(model(input_ids, attention_mask).logits)
        train_metric(output, torch.tensor(label, dtype=torch.int32))

    print(f'Model: {args["model_path"]}')
    print(f'F1: {float(train_metric.compute())}')
    wandb.log({'f1': float(train_metric.compute())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    args = vars(parser.parse_args())

    classes_dict = get_class_dict()
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args['model_path']).to('cuda')

    # MiniLMv2 uses xlm-roberta-large tokenizers but the reference is not present in the config.json, as we
    # downloaded it from Microsoft's GitHub and not HF
    if 'MiniLMv2' in args['model_path']:
        tokenizer = transformers.AutoTokenizer.from_pretrained('xlm-roberta-large')
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args['model_path'])

    wandb.init(config={
        'model': args['model_path'][:-7] + '-acc'
    }, project='dp')

    main()
