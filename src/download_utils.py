import os
import random

import numpy as np
import requests

from global_config import PROJECT_ROOT, CACHE_DIR, INDEX_FILE, INDEX_URL, PREFIX, DATA_DIR

def get_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    return CACHE_DIR


def get_index_file(step_size=1, shuffle=True):
    cache = get_cache_dir()
    filename = os.path.join(cache, INDEX_FILE)
    if not os.path.exists(filename):
        download_file(INDEX_URL, filename)
    ret = [line.strip() for line in open(filename)][::step_size]
    if shuffle:
        random.shuffle(ret)
    return ret


def get_image(timestamp):
    '''
    downloads a ts file and writes the first frame to the cache as a jpeg.

    timestamp is an integer (seconds since unix epoch)
    '''
    file_url = INDEX_URL % (timestamp)
    return file_url


def video_url(file_name):
    return PREFIX + file_name


def image_denormalize(model_output):
    return (model_output * 255).astype(np.uint8)

def download_all_videos(*args):
    data_folder = os.path.join(PROJECT_ROOT, 'data')
    file_list = get_index_file(shuffle=True)
    for this_file in file_list:
        file_name = this_file.split('/')[-1]
        file_path = os.path.join(data_folder, file_name)
        if os.path.exists(file_path):
            continue
        download_file(video_url(this_file), file_path)
        print(this_file, 'download finish')


def download_file_given_file_name(file_name):
    url = video_url(file_name)
    contents = requests.get(url).content
    file_path = os.path.join(DATA_DIR, file_name)
    with open(file_path, 'wb+') as f:
        f.write(contents)

def download_file(url, filename):
    '''
    downloads a the contents of the provided url to a local file
    '''
    contents = requests.get(url).content
    with open(filename, 'wb+') as f:
        f.write(contents)


if __name__ == "__main__":
    from multiprocessing import pool
    p = pool.Pool(20)
    p.map(download_all_videos, [None] * 20)
    # download_all_videos()


