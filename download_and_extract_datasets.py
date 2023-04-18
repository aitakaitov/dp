from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import os
import shutil
import tarfile

import requests


sst_url = 'https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip'
ctdc_url = 'http://ctdc.kiv.zcu.cz/czech_text_document_corpus_v20.tgz'


def download_and_unzip(url, target_dir):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=target_dir)


def download_and_untar(url, target_dir):
    response = requests.get(url, stream=True)
    file = tarfile.open(fileobj=response.raw, mode="r|gz")
    file.extractall(path=target_dir)

def do_sst():
    print('Downloading SST')
    os.makedirs('temp', exist_ok=True)
    download_and_unzip(sst_url, 'temp')
    print('Copying')
    for file in os.listdir('temp/stanfordSentimentTreebank'):
        shutil.move(f'temp/stanfordSentimentTreebank/{file}', 'datasets_ours/sst')

    shutil.rmtree('temp')


def do_ctdc():
    print('Downloading CTDC')
    os.makedirs('temp', exist_ok=True)
    download_and_untar(ctdc_url, 'temp')
    print('Copying')
    shutil.move('temp/czech_text_document_corpus_v20', 'datasets_ours/news')
    shutil.rmtree('temp')


def main():
    do_sst()
    do_ctdc()

if __name__ == '__main__':
    main()