import json
import os


total = 1821.0
lines = ['model;acc;sure;unsure\n']
for directory in sorted(os.listdir('.')):
    if '.csv' in directory or '.py' in directory:
        continue

    sure_attrs_file = directory + '/certain/gradient_attrs.json'
    with open(sure_attrs_file, 'r', encoding='utf-8') as f:
        sure_attrs = json.loads(f.read())

    unsure_attrs_file = directory + '/unsure/gradient_attrs.json'
    with open(unsure_attrs_file, 'r', encoding='utf-8') as f:
        unsure_attrs = json.loads(f.read())

    sure = len(sure_attrs)
    unsure = len(unsure_attrs)

    lines.append(f'{directory[:-6]}l;{(sure + unsure) / total};{sure};{unsure}\n')

with open('acc.csv', 'w+', encoding='utf-8') as f:
    f.writelines(lines)
