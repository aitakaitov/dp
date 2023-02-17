import os
import argparse


def main():
    model_dirs = os.listdir(args['dir'])
    for model_dir in model_dirs:
        if 'attrs' not in model_dir or 'csv' in model_dir:
            continue

        print(f'Processing {model_dir}')

        command = f'python process_attributions_news.py ' \
                  f'--attrs_dir {os.path.join(args["dir"], model_dir)} ' \
                  f'--output_file {os.path.join(args["dir"], model_dir + "-posneg-metrics.csv")}'

        os.system(command)
        
        input()

        command = f'python process_attributions_news.py ' \
                  f'--attrs_dir {os.path.join(args["dir"], model_dir)} ' \
                  f'--output_file {os.path.join(args["dir"], model_dir + "-pos-metrics.csv")} ' \
                  f'--positive_only True'

        os.system(command)
        
        input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    args = vars(parser.parse_args())
    main()
