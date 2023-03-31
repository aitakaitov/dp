import json


file = 'train.csv'

with open(file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('classes.json', 'r', encoding='utf-8') as f:
    classes_37 = json.loads(f.read())

print(f'samples: {len(lines)}')

classes60 = 0
classes37 = 0
tokens = 0

for line in lines:
    split = line.strip().split('~')
    clss = split[1:]
    tokens += len(split[0].split())
    classes60 += len(clss)
    for cls in clss:
        if cls in classes_37.keys():
            classes37 += 1

print(f'classes60: {classes60}')
print(f'classes37: {classes37}')
print(f'tokens: {tokens}')