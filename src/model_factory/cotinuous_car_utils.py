import json
import os

import numpy as np
from bootstrapped import bootstrap as bs, stats_functions as bs_stats

from download_utils import get_index_file
from global_config import FPS, Encoder, SIMILARITY_METRIC, MODEL_DIR, TEST_SIGNIFICANCE_PERCENTILE
from service.restful_global_setting import video_handler, car_model


def is_frame_car(ts, frame_to_skip=FPS):
    try:
        frame = video_handler.get_frame_given_ts(ts, normalize=True, frame_to_skip=frame_to_skip)
        is_car = car_model.predict(
            Encoder.predict(np.array([frame]))
        )[0]
        if (frame is not None) and is_car == 'true':
            return True
    except ValueError as e:
        # convert exception to result
        raise ValueError(e)
    return False


def watch_n_random_videos(n=30, max_samples_per_clip=100, fps=FPS, shuffle=True, max_frame=5000):
    for clip_range in get_n_continuous_car_frame_indices(n, fps=fps, shuffle=shuffle, max_frame=max_frame):
        left, right = clip_range
        length = right-left
        samples_to_draw = min(length, max_samples_per_clip)
        print(left, right)
        l_imgs_ts = np.random.randint(left, right, samples_to_draw)
        r_imgs_ts = np.random.randint(left, right, samples_to_draw)
        try:
            for ts_idx in range(len(l_imgs_ts)):
                l_ts = l_imgs_ts[ts_idx]
                l_t_img = video_handler.get_frame_given_ts(l_ts)
                l_t_emd = Encoder.predict(np.array([l_t_img]))
                r_ts = r_imgs_ts[ts_idx]
                r_t_img = video_handler.get_frame_given_ts(r_ts)
                r_t_emd = Encoder.predict(np.array([r_t_img]))
                diff = SIMILARITY_METRIC(l_t_emd, r_t_emd)[0][0]
                yield diff
        except ValueError:
            continue


def get_n_continuous_car_frame_indices(n=30, fps=FPS, shuffle=True, fast_forward_cap_speed=60, max_frame=5000,
                                       multipler_cap=20):
    file_list = get_index_file(1, shuffle=shuffle,)
    ts_list = [int(file_name.split('.')[0]) for file_name in file_list]
    counter = 1
    buffer = []
    multiplier = 0
    is_prev_frame_car = False
    for ts in ts_list:
        start_ts = ts
        while True:
            try:
                if ts - start_ts > max_frame:
                    if len(buffer) < 2:
                        buffer = []
                        break
                    counter += 1
                    yield (buffer[0], buffer[-1])
                    buffer = []
                    is_prev_frame_car = False
                    multiplier = 0
                    ts += 1
                    break
                if is_frame_car(ts, frame_to_skip=fps):
                    buffer.append(ts)
                    if is_prev_frame_car:
                        multiplier = min(multiplier + 1, multipler_cap)
                        this_step = min(2 ** multiplier, fast_forward_cap_speed)
                        ts += this_step
                        # print(ts, 2 ** multiplier, this_step)
                    else:
                        is_prev_frame_car = True
                        multiplier = 0
                        ts += 1
                else:
                    if is_prev_frame_car:
                        counter += 1
                        yield (buffer[0], buffer[-1])
                        buffer = []
                        is_prev_frame_car = False
                        multiplier = 0
                        ts += 1
                        break
                    else:
                        buffer = []
                        break
            except ValueError:
                ts += 1
                continue

        if counter % n == 0 and counter > 1:
            break


def learn_similar_car_from_videos(num_instances=10, fps=24, learn_new=False,
                                  percentile=5, max_samples_per_clip=30, max_frame=600):
    '''
    Learn similarity distribution from continuous frames that both contains cars
    :param num_instances: number of positive instances need to observe
    :param fps: fps used in video indexing
    :param interval: interval for compare continuous images
    :return: dictionary contains normal distribution mean and std
    '''
    this_dist_path = os.path.join(MODEL_DIR, TEST_SIGNIFICANCE_PERCENTILE)
    if os.path.exists(this_dist_path) and (not learn_new):
        with open(this_dist_path, 'r') as f:
            car_sim_dist = json.load(f)
        return car_sim_dist
    else:
        fps = fps
        n = num_instances
        ret = watch_n_random_videos(n, fps=fps, max_samples_per_clip=max_samples_per_clip, max_frame=max_frame)
        ret = np.array(list(ret))
        print(ret)
        bt_ret = bs.bootstrap(ret, stat_func=bs_stats.mean, alpha=percentile/100)
        print(dir(bt_ret))
        ci = (bt_ret.lower_bound, bt_ret.upper_bound)
        mean = bt_ret.value
        l_percentile, r_percentile = ci
        car_sim_dist = {
            'l_percentile': float(l_percentile),
            'r_percentile': float(r_percentile),
            'mean': float(mean),
        }
        print(car_sim_dist)
        with open(this_dist_path, 'w+') as f:
            json.dump(car_sim_dist, f)
        return car_sim_dist


if __name__ == "__main__":
    learn_similar_car_from_videos(num_instances=30, learn_new=True, max_samples_per_clip=100, max_frame=6000)