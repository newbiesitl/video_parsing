import os

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from download_utils import download_file, video_url
from global_config import ENCODER_PATH, DATA_DIR
from labels.label_creation_pipeline import get_label_file_index, open_video


def build_knn():
    from keras.models import load_model
    encoder = load_model(ENCODER_PATH)
    target_file_path_list = ['true_label_list.txt', 'false_label_list.txt']
    X = []
    Y = []
    for target_file_path in target_file_path_list:
        target_file_list = get_label_file_index(target_file_path)
        # print(true_file_list)
        for label_file in target_file_list:
            file_path = os.path.join(DATA_DIR, label_file)
            if not os.path.exists(file_path):
                download_file(video_url(label_file), file_path)

            for frame in open_video(file_path):
                # print(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img2 = np.zeros_like(frame)
                img2[:, :, 0] = gray
                img2[:, :, 1] = gray
                img2[:, :, 2] = gray
                frame = img2
                embed = encoder.predict(np.array([frame]))[0]
                X.append(embed)
                Y.append(target_file_path)
                # cv2.imwrite(os.path.join(LABEL_FRAME_DIR, label_file+'.png'), frame)
                # cv2.imshow('Frame_focus', frame)
                break
    knn = KNeighborsClassifier(metric='cosine', n_neighbors=5)
    knn.fit(X, Y)
    return knn