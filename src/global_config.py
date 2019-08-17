import os
from keras.models import load_model
# height width format
FRAME_SIZE = (80, 80)
# (y, x) format
ATTENTION_COOR = (180, 180)
channels = 3
INPUT_SHAPE = (FRAME_SIZE[0], FRAME_SIZE[1], channels)
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

MIN_TS = 1538076003
MAX_TS = 1539326113
FPS = 24

ae = load_model(AE_PATH)
encoder = load_model(ENCODER_PATH)
