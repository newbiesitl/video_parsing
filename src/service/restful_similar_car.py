from global_config import Encoder, MIN_TS, \
    ATTENTION_COOR, FRAME_SIZE, SIMILARITY_METRIC
from scipy import stats
from flask import jsonify, make_response, send_file
from flask_restplus import Resource
import numpy as np
import cv2, io

#  fonts on image
from model_factory.cotinuous_car_utils import learn_similar_car_from_videos
from service.restful_global_setting import *
np.random.seed(1238)

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (1, 30)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

global car_similarity_param
car_similarity_param = None



def normal_pdf(x, norm_param):
    m = norm_param['mean']
    var = norm_param['scale']**2
    rvs = stats.norm.rvs(loc=m, scale=var, size=(50, 2))
    return stats.ttest_1samp(rvs, x)[1]

def percentile_ci(x, ci_param):
    l = ci_param['l_percentile']
    if x < l:
        return False
    return True
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
            p_value_threshold = payload.get('pValue', 10)



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

            similarity = SIMILARITY_METRIC(embedded_frame_l, embedded_frame_r).flatten()[0]

            global car_similarity_param
            if car_similarity_param is None:
                car_similarity_param = learn_similar_car_from_videos(num_instances=10, learn_new=False, percentile=p_value_threshold)

            ret = percentile_ci(similarity, car_similarity_param)
            # test_p_value = normal_pdf(similarity, car_similarity_param)
            print(car_similarity_param)
            print(similarity)
            # print(similarity, test_p_value)
            if ret:
                this_result = ret
            else:
                this_result = ret
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
                if (not ret_l ) or (not ret_r):
                    this_result = 'not a car'
                    if (not ret_l):
                        fontScale = 0.5
                        bottomLeftCornerOfText = (1, 30)
                        cv2.putText(original_frame, str(this_result),
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType,
                                    )
                    if (not ret_r):
                        fontScale = 0.5
                        bottomLeftCornerOfText = (80, 30)
                        cv2.putText(original_frame, str(this_result),
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType,
                                    )
                else:
                    fontScale = 0.5
                    bottomLeftCornerOfText = (1, 30)
                    cv2.putText(original_frame, str(this_result),
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
                            'score': float(similarity),
                            'left percentile value': car_similarity_param['l_percentile'],
                            'right percentile value': car_similarity_param['r_percentile'],
                            'mean': car_similarity_param['mean'],
                            'result': str(this_result),
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



