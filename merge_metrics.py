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

    cols = len(lines[0].strip().split(';'))
    return rows, cols


def extract_matrix(csv_file, rows, cols):
    with open(csv_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]

    matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        values = lines[i].strip().split(';')
        for j in range(cols):
            matrix[i][j] = float(values[i])

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
    merged = np.std(matrices_array, axis=0)
    return merged.tolist()

    # for row in range(len(matrices[0])):
    #     for col in range(len(matrices[0][0])):
    #         _sum = 0
    #         for matrix in range(len(matrices)):
    #             _sum += matrices[matrix][row][col]
    #         avg = _sum / len(matrices)


def save_result(merged_matrix, header, methods, output_csv_file):
    with open(output_csv_file, 'w+', encoding='utf-8') as f:
        f.write(';'.join(header) + ';\n')
        for i in range(len(methods)):
            f.write(f'{methods[i]};')
            for j in range(len(merged_matrix[i])):
                f.write(f'{merged_matrix[i][j]};')
            f.write('\n')


def main():
    csv_files = os.listdir(args['dir'])
    rows, cols = get_matrix_size(os.path.join(args['dir'], csv_files[0]))
    header, methods = extract_header_and_methods(os.path.join(args['dir'], csv_files[0]))

    matrices = []
    for csv_file in csv_files:
        matrices.append(extract_matrix(os.path.join(args['dir'], csv_file), rows, cols))

    merged = merge_matrices(matrices)
    save_result(merged, header, methods, args['output_csv_file'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    parser.add_argument('--output_file')
    args = vars(parser.parse_args())
    main()
