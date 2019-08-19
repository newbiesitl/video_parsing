from data_utils import open_video, window
from download_utils import get_index_file, download_file_given_file_name
from sklearn.metrics.pairwise import cosine_similarity
from global_config import DATA_DIR, Encoder, MODEL_DIR, IS_SAME_CAR_DIST_NAME
import os
import numpy as np
from service.restful_master_service import car_model
from scipy.stats import norm
import json

np.random.seed(0)

def watch_n_random_videos(n=20, fps=24):
    file_list = get_index_file(1, shuffle=True,)
    observations = []
    counter = 1
    for each_file in file_list:
        for score in single_video_produce(each_file, fps):
            print(counter)
            counter += 1
            observations.append(score)
            if counter % n == 0:
                break
        if counter % n == 0:
            break
    return observations


def single_video_produce(video_name, fps, step_size=4):
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
    this_dist_path = os.path.join(MODEL_DIR, IS_SAME_CAR_DIST_NAME)
    if os.path.exists(this_dist_path):
        with open(this_dist_path, 'r') as f:
            car_sim_dist = json.load(f)
    else:
        fps = 24
        n = 10
        ret = watch_n_random_videos(n, fps)
        m = np.average(ret)
        var = np.var(ret)
        car_sim_dist = {
            'mean': float(m),
            'variance': float(var),
        }

        with open(this_dist_path, 'w+') as f:
            json.dump(car_sim_dist, f)
        print(car_sim_dist)