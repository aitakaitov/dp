import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('python_cmd', required=False, default=None)
    args = vars(parser.parse_args())

    PYTHON_CLI_NAME = 'python' if args['python_cmd'] is None else args['python_cmd']

    print('Running create_phrase_sentiments.py')
    os.system(f'{PYTHON_CLI_NAME} create_phrase_sentiments.py')

    print('Running create_sentences_tokens.py')
    os.system(f'{PYTHON_CLI_NAME} create_sentences_tokens.py')

    print('Running create_splits_csv.py')
    os.system(f'{PYTHON_CLI_NAME} create_splits_csv.py')

    print('Running extend_train_set.py')
    os.system(f'{PYTHON_CLI_NAME} extend_train_set.py')

    print('Done')