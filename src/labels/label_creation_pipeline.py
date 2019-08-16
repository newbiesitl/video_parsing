from global_config import LABEL_DIR, frame_shape, attention_coor, DATA_DIR
import numpy as np
import os, random
import cv2
def get_label_file_index(label_file_name, step_size=1, shuffle=True):
    filename = os.path.join(LABEL_DIR, label_file_name)
    ret = [line.strip() for line in open(filename)][::step_size]
    if shuffle:
        random.shuffle(ret)
    return ret


def open_video(file_path):
    cap = cv2.VideoCapture(file_path)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    buf = []
    noise_buf = []
    frame_count = 0
    fps = 4
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        counter = 0
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

if __name__ == "__main__":
    true_label_file_name = 'true_label_list.txt'
    false_label_file_name = 'false_label_list.txt'
    target_file_list = get_label_file_index(true_label_file_name)
    target_file_list = get_label_file_index(false_label_file_name)
    # print(true_file_list)
    for label_file in target_file_list:
        file_path = os.path.join(DATA_DIR, label_file)
        for frame in open_video(file_path):
            # print(frame)
            cv2.imshow('Frame_focus', frame)