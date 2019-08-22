from flask import jsonify, make_response, send_file
from flask_restplus import Resource
from global_config import MIN_TS, ORIGINAL_FRAME_SIZE
from restful_service.restful_global_setting import *
import io, cv2


is_same_car_parser = api.parser()
is_same_car_parser.add_argument('ts', type=int, help='time stamp of frame', default=MIN_TS)
is_same_car_parser.add_argument('x', type=int, help='x coordinate', default=0)
is_same_car_parser.add_argument('y', type=int, help='y coordinate', default=0)
is_same_car_parser.add_argument('width', type=int, help='width of image', default=ORIGINAL_FRAME_SIZE[1])
is_same_car_parser.add_argument('height', type=int, help='height of image', default=ORIGINAL_FRAME_SIZE[0])






@ns.route('/ask/get_image')
class RestfulImpl(Resource):
    @ns.doc(description='''
    Retrieve full image 600, 800
    ''', parser=is_same_car_parser)
    @api.expect(is_same_car_parser, validate=True)
    def get(self):
        '''
        Provide a time stamp (in seconds) to detect the object of given bounding box and time stamp
        :return:
        '''
        try:
            payload = is_same_car_parser.parse_args()
            target_object = payload.get('targetObject', 'car')

            init_object_detection_app_backend(target_object)

            model = model_store.get(target_object, None)
            if model is None:
                raise ValueError('object type %s is not supported' % target_object)
            time_stamp = payload.get('timeStamp', MIN_TS)
            x = payload.get('x', 0)
            y = payload.get('y', 0)
            width = payload.get('width', ORIGINAL_FRAME_SIZE[1])
            height = payload.get('height', ORIGINAL_FRAME_SIZE[0])
            frame, downloadable_ts = video_handler.get_frame_given_ts(time_stamp, h=height, w=width, x=x, y=y,
                                                     get_left_most_file_ts=True, normalize=False)
            image_binary = cv2.imencode('.jpg', frame)[1]

            return send_file(
                io.BytesIO(image_binary),
                mimetype='image/jpeg',
                as_attachment=True,
                attachment_filename='%s.jpg' % ('result'))
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

