from global_config import LABEL_DIR, frame_shape, attention_coor
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
            frame = frame.astype('float32') / 255
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


