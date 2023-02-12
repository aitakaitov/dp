import random

import torch
import numpy as np
from torch.nn.functional import one_hot


class NewsDataset(torch.utils.data.Dataset):

    def __init__(self, data: str, tokenizer, classes_dict):
        self.labels = []
        self.texts = []
        self.tokenizer = tokenizer
        self.class_dict = classes_dict
        self.folds = []
        self.mode = 'train'

        for line in data.split('\n'):
            split_line = line.split('~')

            text = split_line[0]
            encoded = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
            self.texts.append(
                [encoded.data['input_ids'], encoded.data['attention_mask']]
            )

            classes = split_line[1:]
            classes = [self.class_dict[clss] for clss in classes]
            label = torch.sum(one_hot(torch.tensor(classes, dtype=torch.long), len(self.class_dict.items())), dim=0)
            self.labels.append(torch.tensor(label, dtype=torch.float32))

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

