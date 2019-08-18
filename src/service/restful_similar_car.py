from data_utils import open_video, window
from download_utils import get_index_file, download_file_given_file_name
from sklearn.metrics.pairwise import cosine_similarity
from global_config import DATA_DIR, Encoder
import os
import numpy as np
from service.restful_master_service import car_model


def watch_n_random_videos(n=10, fps=24):
    file_list = get_index_file(1, shuffle=True,)[:n]
    observations = []
    for each_file in file_list:
        for score in single_video_produce(each_file, fps):
            observations.append(score)


def single_video_produce(video_name, fps, step_size=2):
    file_path = os.path.join(DATA_DIR, video_name)
    if not os.path.exists(file_path):
        download_file_given_file_name(video_name)
    for frame_pair in window(open_video(file_path, frame_to_skip=fps, normalize=True), step_size):

        encoded = Encoder.predict(np.array(frame_pair))
        img1 = encoded[0]
        img2 = encoded[-1]
        if (car_model.predict([img1])[0] == 'true') and (car_model.predict([img2])[0] == 'true'):
            ret = cosine_similarity([img1], [img2]).flatten()
            yield ret[0]

if __name__ == "__main__":
    fps = 24
    n = 10
    watch_n_random_videos(n, fps)