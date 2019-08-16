from global_config import LABEL_DIR, frame_shape, attention_coor, DATA_DIR, LABEL_FRAME_DIR, ENCODER_PATH
import numpy as np
import os, random
import cv2
def get_label_file_index(label_file_name, step_size=1, shuffle=True):
    filename = os.path.join(LABEL_DIR, label_file_name)
    ret = [line.strip() for line in open(filename)][::step_size]
    if shuffle:
        random.shuffle(ret)
    return ret


def open_video(file_path, fps=1):
    cap = cv2.VideoCapture(file_path)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    frame_count = 0
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_count += 1
        if frame_count % fps != 0:
            continue
        if ret == True:
            # normalize frame value
            # frame = frame.astype('float32') / 255
            h, w = frame_shape
            y, x = attention_coor
            cropped_frame = frame[y:y + h, x:x + w]
            yield cropped_frame
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break

def build_knn():
    from keras.models import load_model
    encoder = load_model(ENCODER_PATH)
    target_file_path = ['true_label_list.txt', 'false_label_list.txt'][0]
    target_file_list = get_label_file_index(target_file_path)
    # print(true_file_list)
    for label_file in target_file_list:
        file_path = os.path.join(DATA_DIR, label_file)
        for frame in open_video(file_path):
            # print(frame)
            cv2.imwrite(os.path.join(LABEL_FRAME_DIR, label_file+'.png'), frame)
            break
            # cv2.imshow('Frame_focus', frame)