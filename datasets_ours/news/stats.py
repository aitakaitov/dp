import json
import string

import numpy as np
from datasets_ours.news.stemming import cz_stem


def get_valid_classes():
    with open('classes.json', 'r', encoding='utf-8') as f:
        d = json.loads(f.read())

    return list(d.keys())


def load_pmi_values():
    with open('PMI.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    classes = get_valid_classes()

    values = []
    for i in range(1, len(lines)):
        split = lines[i].split(',')

        # get class from the second column
        clss = split[1].lower()

        # check if the class is actually one we are classifying
        if clss not in classes:
            continue

        pmi = float(split[6])
        values.append({'pmi': pmi, 'clss': clss, 'kw': split[2].lower(), 'joined_count': int(split[0])})

    return values


def filter(data, min_pmi=1, stem=True, no_phrases=True):
    data_by_clss = {}
    kw_by_clss = {}
    for sample in data:
        pmi = sample['pmi']
        kw = sample['kw']
        clss = sample['clss']
        if pmi < min_pmi:
            continue
        if no_phrases:
            if len(kw.split()) > 1:
                continue
        if stem:
            kw = cz_stem(kw)

        if clss in data_by_clss.keys():
            if kw in kw_by_clss[clss]:
                continue
            kw_by_clss[clss].append(kw)
            data_by_clss[clss].append({'kw': kw, 'pmi': pmi, 'clss': clss, 'joined_count': sample['joined_count']})
        else:
            data_by_clss[clss] = [{'kw': kw, 'pmi': pmi, 'clss': clss, 'joined_count': sample['joined_count']}]
            kw_by_clss[clss] = [kw]

    return data_by_clss


def kw_class_stats(data):
    print(f'Class - Keywords statistics\n--------------------------------------------------')
    classes = get_valid_classes()
    count_dict = {}

    for clss in data.keys():
        count_dict[clss] = 0
        for sample in data[clss]:
            count_dict[sample['clss']] += 1

    for clss in classes:
        if clss not in count_dict.keys():
            print(f'Class [{clss}] has 0 keywords')

    print(f'Minimum keywords: {min(list(count_dict.values()))}')
    print(f'Maximum keywords: {max(list(count_dict.values()))}')
    print(f'Average keywords: {sum(list(count_dict.values())) / float(len(count_dict.values()))}')
    print(f'Total keywords: {sum(list(count_dict.values()))}')

    print(f'Per class information:')
    sorted_asc = dict(sorted(count_dict.items(), key=lambda item: item[1]))
    for clss in sorted_asc.keys():
        print(f'Class: {clss}; number of keywords: {sorted_asc[clss]}')

    print()


def get_pmi_stats_from_clss_dict(data):
    print(f'PMI percentile statistics after filtering\n--------------------------------------------------')
    values = []
    for clss in data:
        values.extend([sample['pmi'] for sample in data[clss]])
    values = np.array(values)
    Ps = list(range(5, 100, 5))
    percentiles = [np.percentile(values, P) for P in Ps]

    for i in range(len(percentiles)):
        print(f'Percentile: {Ps[i]}; Value: {percentiles[i]}')

    print()


def get_pmi_stats_from_unfiltered(data):
    print(f'PMI percentile statistics before filtering\n--------------------------------------------------')
    values = [sample['pmi'] for sample in data]
    values = np.array(values)
    Ps = list(range(5, 100, 5))
    percentiles = [np.percentile(values, P) for P in Ps]

    for i in range(len(percentiles)):
        print(f'Percentile: {Ps[i]}; Value: {percentiles[i]}')

    print()


def get_occurrence_stats(data, stemmed=True):
    print(f'Class - keyword occurrences in the test split\n--------------------------------------------------')

    # TODO load the test set and stem all the words
    # TODO check how many keywords for each class are present in each of the documents
    # TODO and print out the results

    class_keywords = { clss: [sample['kw'] for sample in data[clss]] for clss in data.keys() }

    #classes = get_valid_classes()

    with open('test.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    class_texts = {}
    for line in lines:
        split = line.strip().split('~')
        text = split[0].lower().translate(str.maketrans('', '', string.punctuation))
        classes = split[1:]

        text_words = text.split()
        if stemmed:
            text_words = [cz_stem(word) for word in text_words]

        for clss in classes:
            if clss in class_texts.keys():
                class_texts[clss].append(text_words)
            else:
                class_texts[clss] = [text_words]

    occurence_counts = {}   # occurence_count: document_count
    per_class_occurrences = {}
    for clss, texts in class_texts.items():
        per_class_occurrences[clss] = 0

        if clss not in class_keywords.keys():
            continue

        keywords = class_keywords[clss]

        for text in texts:
            if len(text) > 512:
                continue

            present = [1 for kw in keywords if kw in text]

            if len(present) in occurence_counts.keys():
                occurence_counts[len(present)] += 1
            else:
                occurence_counts[len(present)] = 1

            per_class_occurrences[clss] += len(present)

    print(f'Number of keyword occurrences per class:')
    for clss, count in dict(sorted(per_class_occurrences.items(), key=lambda item: item[1])).items():
        print(f'Class: {clss}; Count: {count}')
    print()
    print(f'Number of documents with keyword occurrences:')
    for occ, docs in dict(sorted(occurence_counts.items(), key=lambda item: item[0])).items():
        print(f'Occurences: {occ}; Document count: {docs}')

    print()

def main():
    data = load_pmi_values()
    filtered = filter(data, min_pmi=5, stem=True)

    # get percentiles of PMI on unfiltered data
    get_pmi_stats_from_unfiltered(data)

    # get min, max and avg keywords per class
    kw_class_stats(filtered)

    # get percentiles of PMI on filtered data
    get_pmi_stats_from_clss_dict(filtered)

    # get number of occurrences of each class' keywords in the test set
    get_occurrence_stats(filtered)

    pass


if __name__ == '__main__':
    main()