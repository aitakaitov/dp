import argparse
import os

import transformers
import torch
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import tqdm
from datasets_ours.news.news_dataset import NewsDataset
import json


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')

# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'


def get_class_dict():
    with open('datasets_ours/news/classes.json', 'r', encoding='utf-8') as f:
        class_dict = json.loads(f.read())
    return class_dict


def get_file_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def train(learning_rate, epochs, model):
    train = NewsDataset(get_file_text('datasets_ours/news/train.csv'), tokenizer, classes_dict)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # optimization
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = transformers.AdamW(model.parameters(), lr=learning_rate, eps=1e-08)
    epoch_iters = len(train_dataloader)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=epoch_iters,
                                                             num_training_steps=epoch_iters * epochs * 25)
    sigmoid = torch.nn.Sigmoid()

    # metrics
    train_metric = torchmetrics.F1Score().to(device)
    writer = SummaryWriter(output_dir + '/logs')

    model.to(device)

    # training, eval
    for epoch_num in range(epochs):
        print(f'EPOCH: {epoch_num + 1}')
        iteration = 0
        model.train()
        for train_input, train_label in tqdm.tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = torch.squeeze(train_input[1].to(device), dim=0)
            input_id = train_input[0].squeeze(1).to(device)

            output = model(input_id, mask).logits

            with torch.autocast(device):
                batch_loss = criterion(output, train_label)

            writer.add_scalar('loss/train', float(batch_loss), epoch_num * len(train) + iteration)
            train_metric(sigmoid(output), torch.tensor(train_label, dtype=torch.int32))

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            iteration += 1
            scheduler.step()

        writer.add_scalar('f1/train', train_metric.compute(), (epoch_num + 1) * len(train))
        print(f'F1 TRAIN: {float(train_metric.compute())}')
        train_metric.reset()

        torch.save(model, output_dir + '/model-epoch-' + str(epoch_num + 1))

    model.save_pretrained(output_dir)


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=4, help="Number of training epochs", type=int)
parser.add_argument("--lr", default=1e-5, help="Learning rate", type=float)
parser.add_argument("--model_name", default='UWB-AIR/Czert-B-base-cased', help="Pretrained model path")
parser.add_argument("--model_file", required=True, type=str)
parser.add_argument("--batch_size", default=1, help="Batch size", type=int)
parser.add_argument("--output_dir", default='kfold-training-output', help="Output directory")
parser.add_argument("--from_tf", default='False', type=str, help="If True, imported model is a TensorFlow model. Otherwise the imported model is a PyTorch model.")

args = parser.parse_args()

EPOCHS = args.epochs
LR = args.lr
model_name = args.model_name
batch_size = args.batch_size
output_dir = args.output_dir

# Special case for Czert
from_tf = True if 'Czert' in model_name else args.from_tf.lower() == 'true'

try:
    os.mkdir(output_dir)
except OSError:
    pass

classes_dict = get_class_dict()
model = torch.load(args.model_file)

# MiniLMv2 uses xlm-roberta-large tokenizers but the reference is not present in the config.json, as we
# downloaded it from Microsoft's GitHub and not HF
if 'MiniLMv2' in model_name:
    tokenizer = transformers.AutoTokenizer.from_pretrained('xlm-roberta-large')
else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

train(LR, EPOCHS, model)
