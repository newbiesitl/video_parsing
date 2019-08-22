from flask import jsonify, make_response, send_file
from flask_restplus import Resource
from global_config import MIN_TS, ORIGINAL_FRAME_SIZE, DATA_DIR
from restful_service.restful_global_setting import *
import io,  os


is_same_car_parser = api.parser()
is_same_car_parser.add_argument('ts', type=int, help='time stamp of frame', default=MIN_TS)






@ns.route('/ask/get_file')
class RestfulImpl(Resource):
    @ns.doc(description='''
    Retrieve full image 600, 800
    ''', parser=is_same_car_parser)
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
            _, ts = video_handler.get_closest_file_stream_given_ts(time_stamp)
            file_name = str(ts)+'.ts'
            file_path = os.path.join(DATA_DIR, file_name)


            return send_file(
                file_path,
                as_attachment=True,
                attachment_filename=file_name)
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

