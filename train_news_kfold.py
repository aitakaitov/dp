import os

import torch
import transformers
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import tqdm
from datasets_ours.news.news_dataset import NewsDataset
import json
from sklearn.model_selection import KFold
import random

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')

# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'

import argparse


def get_class_dict():
    with open('datasets_ours/news/classes.json', 'r', encoding='utf-8') as f:
        class_dict = json.loads(f.read())
    return class_dict


def get_file_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def get_fold_sizes(dataset_length, fold_count=5):
    fold_sizes = []
    remaining = dataset_length
    for k in range(fold_count):
        if k < fold_count - 1:
            fold_sizes.append(int(dataset_length / fold_count))
            remaining -= int(dataset_length / fold_count)
        else:
            fold_sizes.append(remaining)
    return fold_sizes


def train(learning_rate, epochs):
    of = open(model_name.replace('/', '_').replace('\\', '_') + '-output', 'w+', encoding='utf-8')

    train = NewsDataset(get_file_text('datasets_ours/news/train.csv'), tokenizer, classes_dict)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # record the eval results
    labels_all = [[] for _ in range(epochs)]
    predictions_all = [[] for _ in range(epochs)]

    fold = 1
    for train_ids, test_ids in kfold.split(train):
        print(f'FOLD {fold}', file=of)
        print('----------------------', file=of)

        # init the fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(train, batch_size=1, sampler=test_subsampler)

        # fresh model
        model = torch.load(BASE_MODEL_PATH).to(device)

        # optimization
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        optimizer = transformers.AdamW(model.parameters(), lr=learning_rate, eps=1e-08)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
        sigmoid = torch.nn.Sigmoid().to(device)

        # metrics
        train_metric = torchmetrics.F1Score().to(device)

        # training, eval
        for epoch_num in range(epochs):
            print(f'EPOCH: {epoch_num + 1}', file=of)
            iteration = 0
            model.train()
            for train_input, train_label in tqdm.tqdm(trainloader):
                train_label = train_label.to(device)
                mask = torch.squeeze(train_input[1].to(device), dim=0)
                input_id = train_input[0].squeeze(1).to(device)

                output = model(input_id, mask).logits

                with torch.autocast(device):
                    batch_loss = criterion(output, train_label)

                train_metric(sigmoid(output), torch.tensor(train_label, dtype=torch.int32))

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                iteration += 1

            print(f'F1 TRAIN: {float(train_metric.compute())}', file=of)
            train_metric.reset()
            model.eval()
            scheduler.step()

            for val_input, val_label in tqdm.tqdm(testloader):
                val_label = torch.unsqueeze(val_label.clone().detach(), dim=-1)
                val_label = val_label.to(device)

                mask = val_input[1].to(device)
                input_id = val_input[0].squeeze(1).to(device)

                with torch.no_grad():
                    output = model(input_id, mask)
                    output = sigmoid(output.logits)
                predictions_all[epoch_num].append(output.to('cpu'))
                labels_all[epoch_num].append(val_label.clone().detach().to('cpu').type(torch.IntTensor))

        fold += 1

    i = 0
    print('\n\n')
    for epoch_predictions, epoch_labels in zip(predictions_all, labels_all):
        eval_metric = torchmetrics.F1Score().to('cpu')
        for prediction, label in zip(epoch_predictions, epoch_labels):
            eval_metric(prediction, label)
        print(f'EPOCH {i + 1} EVAL F1: {eval_metric.compute()}', file=of)
        i += 1


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=4, help="Number of training epochs", type=int)
parser.add_argument("--lr", default=1e-5, help="Learning rate", type=float)
parser.add_argument("--model_name", default='UWB-AIR/Czert-B-base-cased', help="Pretrained model path")
parser.add_argument("--batch_size", default=1, help="Batch size", type=int)
parser.add_argument("--output_dir", default='kfold-training-output', help="Output directory")
parser.add_argument("--from_tf", default='False', type=str, help="If True, imported model is a TensorFlow model. Otherwise the imported model is a PyTorch model.")

args = parser.parse_args()

EPOCHS = args.epochs
LR = args.lr
model_name = args.model_name
batch_size = args.batch_size
output_dir = args.output_dir

# Czert does not have a pytorch_model.bin
from_tf = True if 'Czert' in model_name else args.from_tf.lower() == 'true'

# Avoid conflicts when saving the base model
BASE_MODEL_PATH = model_name.replace('/', '_').replace('\\', '_') + '--' + str(random.randint(0, 1_000_000_000))

classes_dict = get_class_dict()
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(classes_dict), from_tf=from_tf).to('cpu')

# MiniLMv2 uses xlm-roberta-large tokenizers but the reference is not present in the config.json, as we
# downloaded it from Microsoft's GitHub and not HF
if 'MiniLMv2' in model_name:
    tokenizer = transformers.AutoTokenizer.from_pretrained('xlm-roberta-large')
else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

torch.save(model, BASE_MODEL_PATH)

train(LR, EPOCHS)
