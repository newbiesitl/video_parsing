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


def get_index_file():
    cache = get_cache_dir()
    filename = os.path.join(cache, INDEX_FILE)
    if not os.path.exists(filename):
        download_file(INDEX_URL, filename)

    return [line.strip() for line in open(filename)]


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


if __name__ == "__main__":
    import cv2

    data_folder = os.path.join(PROJECT_ROOT, 'data')
    file_list = get_index_file()
    for this_file in file_list:
        print(this_file)
        file_name = this_file.split('/')[-1]
        file_path = os.path.join(data_folder, file_name)
        download_file(video_url(this_file), file_path)
        cap = cv2.VideoCapture(file_path)
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:

                # Display the resulting frame
                cv2.imshow('Frame', frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break
        exit()