import os
import requests

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.join(SCRIPT_PATH, '..', 'cache')
INDEX_FILE = 'index.txt'
INDEX_URL = 'https://hiring.verkada.com/video/index.txt'
PREFIX = 'https://hiring.verkada.com/video/'
TS_URL = '...'
PROJECT_ROOT = SCRIPT_PATH

def get_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    return CACHE_DIR


def get_index_file(step_size=1):
    cache = get_cache_dir()
    filename = os.path.join(cache, INDEX_FILE)
    if not os.path.exists(filename):
        download_file(INDEX_URL, filename)

    return [line.strip() for line in open(filename)][::step_size]


def get_image(timestamp):
    '''
    downloads a ts file and writes the first frame to the cache as a jpeg.

    timestamp is an integer (seconds since unix epoch)
    '''
    file_url = INDEX_URL % (timestamp)
    return file_url


def download_file(url, filename):
    '''
    downloads a the contents of the provided url to a local file
    '''
    contents = requests.get(url).content
    with open(filename, 'wb+') as f:
        f.write(contents)

def video_url(file_name):
    return PREFIX+file_name


def download_all_videos():
    data_folder = os.path.join(PROJECT_ROOT, 'data')
    file_list = get_index_file()
    for this_file in file_list:
        print(this_file)
        file_name = this_file.split('/')[-1]
        file_path = os.path.join(data_folder, file_name)
        download_file(video_url(this_file), file_path)

if __name__ == "__main__":
    # download_all_videos()
    import cv2
    from global_config import frame_shape
    from model_factory.toy_cnn_ae import autoencoder, encoder
    import numpy as np
    from keras.models import load_model
    model_name = 'version_1.m5'
    encoder_name = 'version_1_encoder.m5'
    sample_batch_size = 50
    data_folder = os.path.join(PROJECT_ROOT, 'data')
    file_list = get_index_file()
    encoder_path = os.path.join(PROJECT_ROOT, 'models', encoder_name)
    model_path = os.path.join(PROJECT_ROOT, 'models', model_name)
    if os.path.exists(model_path):
        autoencoder = load_model(model_path)
    for this_file_url in file_list:
        print(this_file_url)
        file_name = this_file_url.split('/')[-1]
        file_path = os.path.join(data_folder, file_name)
        # continue
        if not os.path.exists(file_path):
            download_file(video_url(this_file_url), file_path)

        cap = cv2.VideoCapture(file_path)
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        buf = []
        frame_count = 0
        fps = 24
        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            counter = 0
            frame_count += 1
            if frame_count % fps != 0:
                continue
            if ret == True:
                h, w = frame_shape
                y_sample_idx = np.random.randint(0, frame.shape[0]-h, sample_batch_size)
                x_sample_idx = np.random.randint(0, frame.shape[1]-w, sample_batch_size)
                unioned = zip(y_sample_idx, x_sample_idx)
                for coor in unioned:
                    y, x = coor
                    # x, y = 180, 180
                    cropped_frame = frame[y:y+h, x:x+w]
                    # print(frame.shape)
                    # Display the resulting frame
                    buf.append(cropped_frame)
                    counter += 1
                    # print(counter)
                    if counter == sample_batch_size:
                        break
                    # Press Q on keyboard to  exit
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                cv2.imshow('Frame', frame)

            # Break the loop
            else:
                break
        buf = np.array(buf)
        autoencoder.fit(buf, buf, epochs=30, verbose=2, batch_size=32)
        autoencoder.save(model_path)
        encoder.set_weights(autoencoder.get_weights())
        encoder.save(encoder_path)