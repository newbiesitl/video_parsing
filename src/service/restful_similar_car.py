from data_utils import open_video, window
from download_utils import get_index_file, download_file_given_file_name
from global_config import Encoder, MODEL_DIR, COSINE_IS_SAME_CAR_DIST_NAME, MIN_TS, \
    ATTENTION_COOR, FRAME_SIZE, FPS, SIMILARITY_METRIC
import os
from scipy import stats
import json
from flask import jsonify, make_response, send_file
from flask_restplus import Resource
import numpy as np
import cv2, io

#  fonts on image
from service.restful_global_setting import *
np.random.seed(1238)


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

def watch_n_random_videos(n=30, max_samples_per_clip=100, fps=FPS, shuffle=True):
    clip_ts_range_list = list(get_n_continuous_car_frame_indices(n, fps=fps, shuffle=shuffle))
    for clip_range in clip_ts_range_list:
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
                diff = SIMILARITY_METRIC(l_t_emd, r_t_emd)
                yield diff
        except ValueError:
            continue

def get_n_continuous_car_frame_indices(n=30, fps=FPS, shuffle=True, fast_forward_cap_speed=60, max_frame=5000):
    file_list = get_index_file(1, shuffle=shuffle,)
    ts_list = [int(file_name.split('.')[0]) for file_name in file_list]
    counter = 1
    buffer = []
    multipler = 0
    is_prev_frame_car = False
    for ts in ts_list:
        start_ts = ts
        while True:
            try:
                if ts - start_ts > max_frame:
                    break
                if is_frame_car(ts, frame_to_skip=fps):
                    buffer.append(ts)
                    if is_prev_frame_car:
                        this_step = min(2 ** multipler, fast_forward_cap_speed)
                        print('step size', this_step, multipler)
                        multipler += 1
                        ts += this_step
                    else:
                        is_prev_frame_car = True
                        multipler = 0
                        ts += 1
                else:
                    if is_prev_frame_car:
                        counter += 1
                        yield (buffer[0], buffer[-1])
                        buffer = []
                        is_prev_frame_car = False
                        multipler = 0
                        ts += 1
                        break
                    else:
                        break
            except ValueError:
                ts += 1
                continue

        if counter % n == 0:
            break


def learn_similar_car_from_videos(num_instances=10, fps=24, learn_new=False):
    '''
    Learn similarity distribution from continuous frames that both contains cars
    :param num_instances: number of positive instances need to observe
    :param fps: fps used in video indexing
    :param interval: interval for compare continuous images
    :return: dictionary contains normal distribution mean and std
    '''
    this_dist_path = os.path.join(MODEL_DIR, COSINE_IS_SAME_CAR_DIST_NAME)
    if os.path.exists(this_dist_path) and (not learn_new):
        with open(this_dist_path, 'r') as f:
            car_sim_dist = json.load(f)
        return car_sim_dist
    else:
        fps = fps
        n = num_instances
        ret = watch_n_random_videos(n, fps,)
        ret = list(ret)
        m = np.average(ret)
        var = np.std(ret)
        car_sim_dist = {
            'mean': float(m),
            'scale': float(var),
        }

        with open(this_dist_path, 'w+') as f:
            json.dump(car_sim_dist, f)
        return car_sim_dist




font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (1, 30)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

global car_similarity_norm_param
car_similarity_norm_param = None



def normal_pdf(x, norm_param):
    m = norm_param['mean']
    var = norm_param['scale']
    rvs = stats.norm.rvs(loc=m, scale=var, size=(50, 2))
    return stats.ttest_1samp(rvs, x)[1]



is_same_car_parser = api.parser()
is_same_car_parser.add_argument('targetObject', type=str, choices=['car', ],
                                default='car',
                                help='type of object to detect', )
is_same_car_parser.add_argument('timeStamp1', type=int, help='time stamp of frame 1', default=MIN_TS)
is_same_car_parser.add_argument('timeStamp2', type=int, help='time stamp of frame 2', default=MIN_TS)
is_same_car_parser.add_argument('x', type=int, help='x value',
                                default=ATTENTION_COOR[1])
is_same_car_parser.add_argument('y', type=int,
                                help='y value', default=ATTENTION_COOR[0])
is_same_car_parser.add_argument('width', type=int, choices=[80], default=80,
                                help='width of input image', )
is_same_car_parser.add_argument('height', type=int, choices=[80], default=80,
                                help='height of input image', )
is_same_car_parser.add_argument('returnType', type=str, choices=['image', 'json'], default='image',
                                help='type of response object', )
is_same_car_parser.add_argument('pValue', type=int, choices=[1, 5, 10], default=5,
                                help='p value of is same car test', )






@ns.route('/ask/match')
class ObjDetect(Resource):
    @ns.doc(description='''
    Current detection only support one size image
    ( 80 * 80 )
    ''', parser=is_same_car_parser)
    @api.expect(is_same_car_parser, validate=True)
    def get(self):
        '''
        Check if two frames contain the same car
        :return:
        '''
        try:
            payload = is_same_car_parser.parse_args()
            target_object = payload.get('targetObject', 'car')

            init_object_detection_app_backend(target_object)

            model = model_store.get(target_object, None)
            if model is None:
                raise ValueError('object type %s is not supported' % target_object)
            time_stamp_l = payload.get('timeStamp1', MIN_TS)
            time_stamp_r = payload.get('timeStamp2', MIN_TS)
            x = payload.get('x', ATTENTION_COOR[1])
            y = payload.get('y', ATTENTION_COOR[0])
            width = payload.get('width', FRAME_SIZE[1])
            height = payload.get('height', FRAME_SIZE[0])
            return_type = payload.get('returnType', 'image')
            p_value_threshold = payload.get('pValue', 5)



            frame_l, downloadable_ts_l = video_handler.get_frame_given_ts(time_stamp_l, h=height, w=width, x=x, y=y,
                                                     get_left_most_file_ts=True, normalize=True)
            embedded_frame_l = Encoder.predict(np.array([frame_l]))


            frame_r, downloadable_ts_r = video_handler.get_frame_given_ts(time_stamp_r, h=height, w=width, x=x, y=y,
                                                     get_left_most_file_ts=True, normalize=True)
            embedded_frame_r = Encoder.predict(np.array([frame_r]))
            ret_l = car_model.predict(embedded_frame_l)[0]
            ret_r = car_model.predict(embedded_frame_r)[0]
            ret_l = True if ret_l.lower() == 'true' else False
            ret_r = True if ret_r.lower() == 'true' else False

            # if ret_l is False or ret_r is False:
            if False:
                return make_response(
                    jsonify(
                        {
                            'status': 'success!',
                            'downloadable ts 1': str(time_stamp_l) + '.ts',
                            'downloadable ts 2': str(time_stamp_r) + '.ts',
                            'result': """Car in frame 1: %s-Car in frame 2: %s""" % (('True' if ret_l else 'False'),
                                   ('True' if ret_r else 'False'))
                        },
                        200
                    )
                )
            similarity = SIMILARITY_METRIC(embedded_frame_l, embedded_frame_r).flatten()[0]

            global car_similarity_norm_param
            if car_similarity_norm_param is None:
                car_similarity_norm_param = learn_similar_car_from_videos(num_instances=10, learn_new=False)
            test_p_value = normal_pdf(similarity, car_similarity_norm_param)
            print(car_similarity_norm_param)
            print(similarity, test_p_value)
            if test_p_value[0] < p_value_threshold/100:
                this_result = 'false'
            else:
                this_result = 'true'
            if return_type.lower() == 'image':
                original_frame_l, downloadable_ts_l = video_handler.get_frame_given_ts(time_stamp_l, h=height, w=width, x=x,
                                                                                   y=y,
                                                                                   get_left_most_file_ts=True,
                                                                                   normalize=False)
                original_frame_r, downloadable_ts_r = video_handler.get_frame_given_ts(time_stamp_r, h=height, w=width, x=x,
                                                                                   y=y,
                                                                                   get_left_most_file_ts=True,
                                                                                   normalize=False)
                original_frame = np.concatenate((original_frame_l, original_frame_r), axis=1)
                cv2.putText(original_frame, this_result,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType,
                            )
                image_binary = cv2.imencode('.jpg', original_frame)[1]
                return send_file(
                    io.BytesIO(image_binary),
                    mimetype='image/jpeg',
                    as_attachment=True,
                    attachment_filename='%s.jpg' % ('result'))
            else:
                return make_response(
                    jsonify(
                        {
                            'status': 'success!',
                            'downloadable ts 1': str(downloadable_ts_l)+'.ts',
                            'downloadable ts 2': str(downloadable_ts_r)+'.ts',
                            'left p value': test_p_value[0],
                            'right p value': test_p_value[1],
                            'result': this_result,
                        },
                        200
                    )
                )

        except ValueError as e:
            return make_response(
                jsonify(
                    {
                        'status': 'error',
                        'details': str(e),
                    }
                ),
                403
            )




if __name__ == "__main__":
    learn_similar_car_from_videos(30, learn_new=True)