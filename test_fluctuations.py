import torch
import transformers
from models.bert_512 import BertForSequenceClassificationChefer
from attribution_methods_custom import __ig_interpolate_samples, gradient_attributions

model_path = 'bert-base-cased-sst-v2-1'

model = BertForSequenceClassificationChefer.from_pretrained(model_path)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model.to('cuda')
model.eval()
embeddings = model.bert.embeddings.word_embeddings.weight

sample_input = 'If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .'
encoded = tokenizer(sample_input, return_tensors='pt')
inputs_embeds = torch.index_select(embeddings, dim=0, index=encoded['input_ids'][0].to('cuda'))

# deal with smoothgrad
offset = (inputs_embeds * 0) + (torch.max(inputs_embeds) - torch.min(inputs_embeds)) * 0.05
baseline = inputs_embeds - offset
target = inputs_embeds + offset
sg_samples = __ig_interpolate_samples(baseline, target, 1000)

grads_target = gradient_attributions(torch.unsqueeze(inputs_embeds, dim=0).to('cuda'), encoded['attention_mask'].to('cuda'), 1, model, torch.nn.Softmax(dim=-1)).to('cpu')
value_target = torch.sum(grads_target[0], dim=1)


values = []
maximums = []
for sample in sg_samples:
    continue
    sample = torch.unsqueeze(sample, dim=0).to('cuda')
    grads = gradient_attributions(sample, encoded['attention_mask'].to('cuda'), 1, model).to('cpu')
    grads = torch.sum(grads[0], dim=1)
    #maximums.append(torch.max(grads[-6]).item())
    values.append(torch.mean(grads - value_target))

with open('point_gradients.csv', 'w+', encoding='utf-8') as f:
    f.write('sample;gradient;\n')
    for i, value in enumerate(values):
        f.write(f'{i};{value};\n')


baseline = inputs_embeds * 0
ig_samples = __ig_interpolate_samples(baseline, inputs_embeds, 100)

sums = []
for sample in ig_samples:
    sample = torch.unsqueeze(sample, dim=0).to('cuda')
    grads = gradient_attributions(sample, encoded['attention_mask'].to('cuda'), 1, model, torch.nn.Softmax(dim=-1)).to('cpu')
    sums.append(torch.sum(torch.abs(grads)))

with open('absolute_gradients.csv', 'w+', encoding='utf-8') as f:
    f.write('step;gradient sum;\n')
    for i, _sum in enumerate(sums):
        f.write(f'{i};{_sum};\n')


