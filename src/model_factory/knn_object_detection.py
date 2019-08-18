import os

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from download_utils import download_file, get_video_url
from global_config import DATA_DIR, Encoder
from data_utils import get_label_file_index, open_video


def build_knn():
    target_file_path_list = ['true.txt', 'false.txt']
    X = []
    Y = []
    for target_file_path in target_file_path_list:
        target_file_list = get_label_file_index(target_file_path)
        # print(true_file_list)
        for label_file in target_file_list:
            file_path = os.path.join(DATA_DIR, label_file)
            if not os.path.exists(file_path):
                download_file(get_video_url(label_file), file_path)

            for frame in open_video(file_path):
                # print(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img2 = np.zeros_like(frame)
                img2[:, :, 0] = gray
                img2[:, :, 1] = gray
                img2[:, :, 2] = gray
                frame = img2
                embed = Encoder.predict(np.array([frame]))[0]
                X.append(embed)
                label = target_file_path.split('.')[0]
                Y.append(label)
                # cv2.imwrite(os.path.join(LABEL_FRAME_DIR, label_file+'.png'), frame)
                # cv2.imshow('Frame_focus', frame)
                break
    knn = KNeighborsClassifier(metric='cosine', n_neighbors=5)
    knn.fit(X, Y)
    return knn