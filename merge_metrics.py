import argparse
import os
import numpy as np


def get_matrix_size(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]

    rows = 0
    for line in lines:
        if line.strip() != '':
            rows += 1

    cols = len(lines[0].strip().split(';')) - 1
    return rows, cols


def extract_matrix(csv_file, rows, cols):
    with open(csv_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]

    matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        values = lines[i].strip().split(';')[1:]
        for j in range(cols):
            matrix[i][j] = float(values[j])

    return matrix


def extract_header_and_methods(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header = lines[0].strip().split(';')
    methods = []
    for line in lines[1:]:
        if line.strip() == '':
            continue
        methods.append(line.split(';')[0])

    return header, methods


def merge_matrices(matrices):
    matrices_array = np.array(matrices)
    stdev = np.std(matrices_array, axis=0)
    avg = np.mean(matrices_array, axis=0)
    return avg.tolist(), stdev.tolist()


def save_result(average_matrix, stdev_matrix, header, methods, output_csv_file):
    with open(output_csv_file, 'w+', encoding='utf-8') as f:
        f.write(';'.join(header) + ';\n')
        for i in range(len(methods)):
            f.write(f'{methods[i]};')
            for j in range(len(average_matrix[i])):
                f.write(f'{average_matrix[i][j]:.3f} ± {stdev_matrix[i][j]:.3f};')
            f.write('\n')


def main():
    csv_files = os.listdir(args['dir'])
    csv_files = [f for f in csv_files if 'csv' in f]
    rows, cols = get_matrix_size(os.path.join(args['dir'], csv_files[0]))
    header, methods = extract_header_and_methods(os.path.join(args['dir'], csv_files[0]))

    matrices = []
    for csv_file in csv_files:
        matrices.append(extract_matrix(os.path.join(args['dir'], csv_file), rows, cols))

    average, stdev = merge_matrices(matrices)
    save_result(average, stdev, header, methods, args['output_file'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    parser.add_argument('--output_file')
    args = vars(parser.parse_args())
    print(args)
    main()
