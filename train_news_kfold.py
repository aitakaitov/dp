import torch
import transformers
import torchmetrics
from datasets_ours.news.news_dataset import NewsDataset
import json
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
import random
import argparse

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


def train():
    of = open(args['model_name'].replace('/', '_').replace('\\', '_') + f'-{random_number}-output', 'w+',
              encoding='utf-8')

    train_set = NewsDataset(get_file_text('datasets_ours/news/train.csv'), tokenizer, classes_dict)

    # use 5 folds as the dataset's authors
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # record the eval results
    labels_all = [[] for _ in range(args['epochs'])]
    predictions_all = [[] for _ in range(args['epochs'])]

    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_set.input_ids)):
        print(f'FOLD {fold}', file=of)
        print('----------------------', file=of)

        train_input_ids, train_attention_masks, train_labels = train_set.input_ids[train_ids], \
                                                               train_set.attention_masks[train_ids], \
                                                               train_set.labels[train_ids]

        val_input_ids, val_attention_masks, val_labels = train_set.input_ids[test_ids],\
                                                         train_set.attention_masks[test_ids],\
                                                         train_set.labels[test_ids]

        # init the fold
        train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        val_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)
        train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

        # fresh model
        model = torch.load(BASE_MODEL_PATH).to(device)

        # optimization
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        optimizer = transformers.AdamW(model.parameters(), lr=args['lr'], eps=1e-08)

        epoch_iters = len(train_loader)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=epoch_iters,
                                                                 num_training_steps=epoch_iters * args['epochs'] * 25)
        sigmoid = torch.nn.Sigmoid().to(device)

        # metrics
        train_metric = torchmetrics.F1Score().to(device)

        # wandb
        wandb.init(reinit=True, entity='aitakaitov', config={'fold': fold})

        # training, eval
        for epoch_num in range(args['epochs']):
            print(f'EPOCH: {epoch_num + 1}', file=of)
            iteration = 0
            model.train()
            train_loss = 0
            for batch in train_loader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                output = model(b_input_ids, b_input_mask).logits

                with torch.autocast(device):
                    batch_loss = criterion(output, b_labels)

                train_metric(sigmoid(output), torch.tensor(b_labels, dtype=torch.int32))

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                iteration += 1

                train_loss += torch.sum(batch_loss).item()

                scheduler.step()

            print(f'F1 TRAIN: {float(train_metric.compute())}', file=of)
            model.eval()

            test_loss = 0
            for batch in val_loader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                with torch.no_grad():
                    output = model(b_input_ids, b_input_mask).logits
                    batch_loss = criterion(output, b_labels)
                    output = sigmoid(output)

                predictions_all[epoch_num].append(output.to('cpu'))
                labels_all[epoch_num].append(b_labels.clone().detach().to('cpu').type(torch.IntTensor))
                test_loss += batch_loss.item()

            wandb.log({'f1': float(train_metric.compute()),
                       'train_loss': train_loss / len(train_loader) * args['batch_size'],
                       'test_loss': test_loss / len(val_loader)})

            train_metric.reset()

        fold += 1

    i = 0
    print('\n\n')
    for epoch_predictions, epoch_labels in zip(predictions_all, labels_all):
        eval_metric = torchmetrics.F1Score().to('cpu')
        for prediction, label in zip(epoch_predictions, epoch_labels):
            eval_metric(prediction, label)
        print(f'EPOCH {i + 1} EVAL F1: {eval_metric.compute()}', file=of)
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=4, help="Number of training epochs", type=int)
    parser.add_argument("--lr", default=1e-5, help="Learning rate", type=float)
    parser.add_argument("--model_name", default='UWB-AIR/Czert-B-base-cased', help="Pretrained model path")
    parser.add_argument("--batch_size", default=1, help="Batch size", type=int)
    parser.add_argument("--output_dir", default='kfold-training-output', help="Output directory")
    parser.add_argument("--from_tf", default='False', type=str, help="If True, imported model is a TensorFlow model."
                                                                     " Otherwise the imported model is a PyTorch model.")

    args = vars(parser.parse_args())

    # Czert does not have a pytorch_model.bin
    from_tf = True if 'Czert' in args['model_name'] else args['from_tf'].lower() == 'true'

    # Avoid conflicts when saving the base model
    random_number = str(random.randint(0, 1_000_000_000))
    BASE_MODEL_PATH = args['model_name'].replace('/', '_').replace('\\', '_') + '--' + random_number

    classes_dict = get_class_dict()
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args['model_name'],
                                                                            num_labels=len(classes_dict),
                                                                            from_tf=from_tf).to('cpu')

    # MiniLMv2 uses xlm-roberta-large tokenizers but the reference is not present in the config.json, as we
    # downloaded it from Microsoft's GitHub and not HF
    if 'MiniLMv2' in args['model_name']:
        tokenizer = transformers.AutoTokenizer.from_pretrained('xlm-roberta-large')
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args['model_name'])

    # save the initialized model
    torch.save(model, BASE_MODEL_PATH)

    train()
