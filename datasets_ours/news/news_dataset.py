import random

import torch
import numpy as np
from torch.nn.functional import one_hot


class NewsDataset:

    def __init__(self, data: str, tokenizer, classes_dict):
        self.labels = []
        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = tokenizer
        self.class_dict = classes_dict

        for line in data.split('\n')[:100]:
            split_line = line.split('~')

            text = split_line[0]
            encoded = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
            self.input_ids.append(encoded.data['input_ids'])
            self.attention_masks.append(encoded.data['attention_mask'])

            classes = split_line[1:]
            classes = [self.class_dict[clss] for clss in classes]
            label = torch.sum(one_hot(torch.tensor(classes, dtype=torch.long), len(self.class_dict.items())), dim=0)
            self.labels.append(list(label))

        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_masks = torch.cat(self.attention_masks, dim=0)
        self.labels = torch.tensor(self.labels, dtype=torch.float)

