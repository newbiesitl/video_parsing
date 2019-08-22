from flask import jsonify, make_response, send_file
from flask_restplus import Resource
from global_config import MIN_TS, ATTENTION_COOR, FRAME_SIZE, Encoder
import numpy as np
import cv2, io

#  fonts on image
from restful_service.restful_global_setting import *

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (1, 30)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2






obj_detect_parser = api.parser()
obj_detect_parser.add_argument('targetObject', type=str, choices=['car',],
                               default='car',
                               help='type of object to detect', )
obj_detect_parser.add_argument('timeStamp', type=int, help='time stamp of video', default=MIN_TS)
obj_detect_parser.add_argument('x', type=int, help='Reload labels', default=ATTENTION_COOR[1])
obj_detect_parser.add_argument('y', type=int,
                               help='Reload labels', default=ATTENTION_COOR[0])
obj_detect_parser.add_argument('width', type=int, choices=[80], default=80,
                               help='width of input image', )
obj_detect_parser.add_argument('height', type=int, choices=[80], default=80,
                               help='height of input image', )
obj_detect_parser.add_argument('returnType', type=str, choices=['image', 'json'], default='image',
                               help='type of response object', )





@ns.route('/ask/contain')
class RestfulImpl(Resource):
    @ns.doc(description='''
    Current detection only support one size image
    ( 80 * 80 )
    ''', parser=obj_detect_parser)
    @api.expect(obj_detect_parser, validate=True)
    def get(self):
        '''
        Provide a time stamp (in seconds) to detect the object of given bounding box and time stamp
        :return:
        '''
        try:
            payload = obj_detect_parser.parse_args()
            target_object = payload.get('targetObject', 'car')

            init_object_detection_app_backend(target_object)

            model = model_store.get(target_object, None)
            if model is None:
                raise ValueError('object type %s is not supported' % target_object)
            time_stamp = payload.get('timeStamp', MIN_TS)
            x = payload.get('x', ATTENTION_COOR[1])
            y = payload.get('y', ATTENTION_COOR[0])
            width = payload.get('width', FRAME_SIZE[1])
            height = payload.get('height', FRAME_SIZE[0])
            return_type = payload.get('returnType', 'image')
            frame, downloadable_ts = video_handler.get_frame_given_ts(time_stamp, h=height, w=width, x=x, y=y,
                                                     get_left_most_file_ts=True, normalize=True)

            embedded_frame = Encoder.predict(np.array([frame]))
            ret = car_model.predict(embedded_frame)
            this_result = ret[0]


            if return_type.lower() == 'image':
                original_frame, downloadable_ts = video_handler.get_frame_given_ts(time_stamp, h=height, w=width, x=x,
                                                                                   y=y,
                                                                                   get_left_most_file_ts=True,
                                                                                   normalize=False)
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
                            'downloadable ts': str(downloadable_ts)+'.ts',
                            'result': this_result
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





