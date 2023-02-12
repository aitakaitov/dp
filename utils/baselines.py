import os
import torch


def zero_embedding_baseline(inputs_embeds):
    return 0 * inputs_embeds


def pad_baseline(inputs_embeds, pad_embedding):
    # repeat pad embedding based on the number of embedded tokens
    return pad_embedding.repeat(1, inputs_embeds.shape[1], 1)


def embedding_space_average_baseline(inputs_embeds, average_embedding):
    return average_embedding.repeat(1, inputs_embeds.shape[1], 1)


def prepared_baseline(inputs_embeds, baseline_dir):
    baseline = torch.load(os.path.join(baseline_dir, f'{inputs_embeds.shape[1]}.pt'))
    return baseline