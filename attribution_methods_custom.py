import numpy
import torch
from captum.attr import KernelShap


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)


def kernel_shap_attributions(input_ids, attention_mask, target_idx, model, baseline_idx, cls_tensor, sep_tensor,
                             logit_fn, steps=50):
    """
    Generates KernelShap attributions using the Captum implementation
    :param input_ids: input ids - without CLS and SEP tokens
    :param attention_mask: attention mask
    :param target_idx: target index in the model output
    :param model: model
    :param baseline_idx: token to use as a baseline
    :param cls_tensor: a shape (1, 1) tensor with CLS token index
    :param sep_tensor: a shape (1, 1) tensor with SEP token index
    :param logit_fn: function to apply to the logits (e.g. softmax or sigmoid) - needs to be callable
    :param steps: number of pertubed samples to use
    :return:
    """
    # As an exception, we expect the input_ids not to contain CLS and SEP tokens - we have no control of the
    # pertubations done and we need the CLS token to stay there as it's a special case.
    # As such, we pass only the token embeddings to captum and in the proxy function
    # we add the CLS and SEP tokens to the input.

    def f(inputs, att_m):
        # we use this function as a proxy to the model due to how the captum api works
        # captum will squeeze the attention mask, so we give it one more dimension
        t1 = torch.cat((cls_tensor, inputs), dim=1)
        t2 = torch.cat((t1, sep_tensor), dim=1)
        output = model(input_ids=t2, attention_mask=torch.unsqueeze(att_m, dim=0)).logits

        # apply softmax or sigmoid on the logits
        return logit_fn(output)

    ks = KernelShap(f)
    res = ks.attribute(inputs=input_ids, additional_forward_args=attention_mask, target=target_idx,
                       n_samples=steps, baselines=baseline_idx)

    # add dummy values for the removed CLS and SEP tokens to preserve the interface
    res = torch.cat((torch.tensor([[0.0]]).to(device), res), dim=1)
    res = torch.cat((res, torch.tensor([[0.0]]).to(device)), dim=1)

    return res


def gradient_attributions(inputs_embeds, attention_mask, target_idx, model, logit_fn, x_inputs=False):
    """
    Vanilla Gradients
    :param inputs_embeds: input embeddings
    :param attention_mask: attention mask
    :param target_idx: target index in the model output
    :param model: model
    :param x_inputs: multiply by inputs
    :return:
    """
    inputs_embeds = inputs_embeds.requires_grad_(True).to(device)
    attention_mask = attention_mask.to(device)

    model.zero_grad()
    output = logit_fn(model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits)[:, target_idx]
    grads = torch.autograd.grad(output, inputs_embeds)[0]

    if x_inputs:
        grads = grads * inputs_embeds

    return grads


def ig_attributions(inputs_embeds, attention_mask, target_idx, baseline, model, logit_fn, steps=50, method='trapezoid'):
    """
    Generates Integrated Gradients attributions for a sample
    :param inputs_embeds: input embeddings
    :param attention_mask: attention mask
    :param target_idx: taget index in the model output
    :param baseline: what baseline to use as a starting point for the interpolation
    :param model: model
    :param steps: number of interpolation steps
    :return:
    """

    if method == 'trapezoid':
        interpolated_samples = __ig_interpolate_samples(baseline, inputs_embeds, steps)
        gradients = torch.tensor([])
        for sample in interpolated_samples:
            sample = sample.to(device)
            grads = gradient_attributions(sample, attention_mask, target_idx, model, logit_fn).to('cpu')
            gradients = torch.cat((gradients, grads), dim=0)

        gradients = (gradients[:-1] + gradients[1:]) / 2.0
        average_gradients = torch.mean(gradients, dim=0)
        integrated_gradients = (inputs_embeds - baseline) * average_gradients.to(device)

        return integrated_gradients
    else:
        # scale the [-1, 1] interval to [0, 1]
        weights = list(0.5 * numpy.polynomial.legendre.leggauss(steps)[1])
        alphas = list(0.5 * (1 + numpy.polynomial.legendre.leggauss(steps)[0]))

        interpolated_samples = [(baseline + alpha * (inputs_embeds - baseline)).to('cpu') for alpha in alphas]
        total_grads = 0
        for i, sample in enumerate(interpolated_samples):
            sample = sample.to(device)
            grads = gradient_attributions(sample, attention_mask, target_idx, model, logit_fn).to('cpu')
            total_grads += grads * weights[i]

        integrated_gradients = (inputs_embeds - baseline) * total_grads.to(device)
        return integrated_gradients


def __ig_interpolate_samples(baseline, target, steps):
    return [(baseline + (float(i) / steps) * (target - baseline)).to('cpu') for i in range(0, steps + 1)]


def sg_attributions(inputs_embeds, attention_mask, target_idx, model, logit_fn, samples=50, stdev_spread=0.15):
    """
    Generates SmoothGRAD attributions for a sample
    :param inputs_embeds: Input embeddings
    :param attention_mask: attention mask
    :param target_idx: the target index in the model output
    :param model: model
    :param samples: number of noisy samples
    :param stdev_spread: the noise level
    :return:
    """
    stdev = (torch.max(inputs_embeds) - torch.min(inputs_embeds)) * stdev_spread
    length = attention_mask.shape[1]
    samples = __sg_generate_samples(inputs_embeds, length, stdev, samples)
    gradients = torch.tensor([]).to('cpu')
    for sample in samples:
        sample = sample.to(device)
        grads = gradient_attributions(sample, attention_mask, target_idx, model, logit_fn).to('cpu')
        gradients = torch.cat((gradients, grads), dim=0)

    average_gradients = torch.mean(gradients, dim=0)
    return torch.unsqueeze(average_gradients, dim=0)


def __sg_generate_samples(inputs_embeds, length, stdev, samples):
    # Generate noisy samples
    noisy_samples = []
    for i in range(samples):
        means = torch.zeros((1, length, inputs_embeds.shape[2])).to('cpu')
        stdevs = torch.full((1, length, inputs_embeds.shape[2]), float(stdev)).to('cpu')
        noise = torch.normal(means, stdevs).to('cpu')
        sample = inputs_embeds.to('cpu') + noise
        noisy_samples.append(sample)

    return noisy_samples

