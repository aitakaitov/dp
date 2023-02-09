import os
import argparse


def main(args):
    dirs = os.listdir(args['models_dir'])

    for _dir in dirs:
        dir_path = os.path.join(args['models_dir'], _dir)
        tokenizer = dir_path
        output_dir = os.path.join(dir_path, args['baselines_dir_name'])
        os.system(f'python generate_neutral_baselines_random_sst.py --tokenizer {tokenizer} --model_folder {dir_path} --output_dir {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, required=True, help='Directory with models that can be loaded with'
                                                                      ' from_pretrained including the tokenizer')
    parser.add_argument('--baselines_dir_name', type=str, default='baselines-sst', help='Directory in which the baselines'
                                                                                        'will be saved (appended to the '
                                                                                        'specific model directory)')
    args = vars(parser.parse_args())
    main(args)
