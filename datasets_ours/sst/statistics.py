file = 'test.csv'
with open(file, 'r', encoding='utf-8') as f:
    lines = f.readlines()[1:]

print(f'total samples: {len(lines)}')

sentiment = [0, 0]
tokens = 0

for line in lines:
    if line.strip() == '':
        continue
    text, label = line.strip().split('\t')
    sentiment[int(label)] += 1
    tokens += len(text.split())

print(f'positive: {sentiment[1]}')
print(f'negative: {sentiment[0]}')
print(f'tokens: {tokens}')