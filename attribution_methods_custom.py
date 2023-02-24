import torch
from captum.attr import KernelShap


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)


def kernel_shap_attributions(input_ids, attention_mask, target_idx, model, baseline_idx, cls_tensor, sep_tensor,
                             logit_fn, steps=50):
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

        # apply softmax or sigmoid on the logits - the models we use are modified for Chefer et al. and are not
        # compatible with captum. We have to use ModelForSequenceClassification which provides logits
        return logit_fn(output)

    ks = KernelShap(f)
    res = ks.attribute(inputs=input_ids, additional_forward_args=attention_mask, target=target_idx,
                       n_samples=steps, baselines=baseline_idx)

    # add dummy values for the removed CLS and SEP tokens to preserve the interface
    res = torch.cat((torch.tensor([[0.0]]).to(device), res), dim=1)
    res = torch.cat((res, torch.tensor([[0.0]]).to(device)), dim=1)

    return res


def gradient_attributions(input_embeds, attention_mask, target_idx, model, x_inputs=False):
    input_embeds = input_embeds.requires_grad_(True).to(device)
    attention_mask = attention_mask.to(device)

    model.zero_grad()
    output = model(input_embeds, attention_mask=attention_mask)[:, target_idx]
    grads = torch.autograd.grad(output, input_embeds)[0]

    if x_inputs:
        grads = grads * input_embeds

    return grads


def ig_attributions(inputs_embeds, attention_mask, target_idx, baseline, model, steps=50):
    # scale inputs and compute gradients
    interpolated_samples = __ig_interpolate_samples(baseline, inputs_embeds, steps)
    gradients = torch.tensor([])
    for sample in interpolated_samples:
        sample = sample.to(device)
        grads = gradient_attributions(sample, attention_mask, target_idx, model).to('cpu')
        gradients = torch.cat((gradients, grads), dim=0)

    gradients = (gradients[:-1] + gradients[1:]) / 2.0
    average_gradients = torch.mean(gradients, dim=0)
    integrated_gradients = (inputs_embeds - baseline) * average_gradients.to(device)
    return integrated_gradients


def __ig_interpolate_samples(baseline, target, steps):
    return [(baseline + (float(i) / steps) * (target - baseline)).to('cpu') for i in range(0, steps + 1)]


def sg_attributions(inputs_embeds, attention_mask, target_idx, model, samples=50, stdev_spread=0.15):
    stdev = (torch.max(inputs_embeds) - torch.min(inputs_embeds)) * stdev_spread
    length = attention_mask.shape[1]
    samples = __sg_generate_samples(inputs_embeds, length, stdev, samples)
    gradients = torch.tensor([]).to('cpu')
    for sample in samples:
        sample = sample.to(device)
        grads = gradient_attributions(sample, attention_mask, target_idx, model).to('cpu')
        gradients = torch.cat((gradients, grads), dim=0)

    average_gradients = torch.mean(gradients, dim=0)
    return torch.unsqueeze(average_gradients, dim=0)


def __sg_generate_samples(inputs_embeds, length, stdev, samples):
    noisy_samples = []
    for i in range(samples):
        means = torch.zeros((1, length, inputs_embeds.shape[2])).to('cpu')
        stdevs = torch.full((1, length, inputs_embeds.shape[2]), float(stdev)).to('cpu')
        noise = torch.normal(means, stdevs).to('cpu')
        sample = inputs_embeds.to('cpu') + noise
        noisy_samples.append(sample)

    return noisy_samples

