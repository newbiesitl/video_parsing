import os

frame_shape = (80, 80)
attention_coor = (180, 180)
channels = 3
input_shape = (frame_shape[0], frame_shape[1], channels)
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
LABEL_DIR = os.path.join(SCRIPT_PATH, 'labels')
LABEL_FRAME_DIR = os.path.join(LABEL_DIR, 'frames')
CACHE_DIR = os.path.join(SCRIPT_PATH, '..', 'cache')
DATA_DIR = os.path.join(SCRIPT_PATH, 'data')
MODEL_DIR = os.path.join(SCRIPT_PATH, 'models')
INDEX_FILE = 'index.txt'
INDEX_URL = 'https://hiring.verkada.com/video/index.txt'
PREFIX = 'https://hiring.verkada.com/video/'
TS_URL = '...'
PROJECT_ROOT = SCRIPT_PATH
model_name = 'version_1.m5'
encoder_name = 'version_1_encoder.m5'
AE_PATH = os.path.join(MODEL_DIR, model_name)
ENCODER_PATH = os.path.join(MODEL_DIR, encoder_name)