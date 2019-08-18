from data_utils import open_video, window
from sklearn.metrics.pairwise import cosine_similarity
from global_config import DATA_DIR, Encoder
import os
import numpy as np
import requests




def single_video_produce(video_name, fps, step_size=2):
    file_path = os.path.join(DATA_DIR, video_name)

    for frame_pair in window(open_video(file_path, frame_to_skip=fps), step_size):

        encoded = Encoder.predict(np.array(frame_pair))
        ret = cosine_similarity([encoded[0]], [encoded[-1]]).flatten()










if __name__ == "__main__":
    fps = 24
    video_name = '1538076003.ts'
    single_video_produce(video_name, fps)