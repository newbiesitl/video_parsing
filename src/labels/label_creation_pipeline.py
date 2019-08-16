from global_config import LABEL_DIR
import os, random

def get_label_file_index(label_file_name, step_size=1, shuffle=True):
    filename = os.path.join(LABEL_DIR, label_file_name)
    ret = [line.strip() for line in open(filename)][::step_size]
    if shuffle:
        random.shuffle(ret)
    return ret