import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str)
parser.add_argument('--output', default='model_pretrained', type=str)
args = vars(parser.parse_args())

model = torch.load(args['target'])
model.save_pretrained(args['output'])