import os
import argparse


def main():
    files = os.listdir(args['dir'])
    for file in files:
        if 'attrs' not in file:
            continue

        command = f'python process_attributions_sst.py --attrs_dir {os.path.join(args["dir"], file)} ' \
                  f'--output_file {os.path.join(args["dir"], file + "-posneg-metrics.csv")}'

        os.system(command)

        command = f'python process_attributions_sst.py --attrs_dir {os.path.join(args["dir"], file)} ' \
                  f'--output_file {os.path.join(args["dir"], file + "-pos-metrics.csv")} ' \
                  f'--positive_only True'

        os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    args = vars(parser.parse_args())
    main()
