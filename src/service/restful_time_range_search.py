from flask import Flask, jsonify, make_response, Blueprint
from flask_restplus import Api, Resource, fields
from model_factory.knn_object_detection import  build_knn
from download_utils import get_index_file
from global_config import MIN_TS, ATTENTION_COOR, FRAME_SIZE, encoder, ae
import numpy as np




# init app
app = Flask(__name__)
app.config.SWAGGER_UI_DOC_EXPANSION = 'full'
blueprint = Blueprint('api', __name__, url_prefix='')
api = Api(blueprint, version='1.0',title='Stream plus', description='Content indexing')
app.register_blueprint(blueprint)
ns = api.namespace('v1', description='Content parsing node')


obj_detect_parser = api.parser()
obj_detect_parser.add_argument('target object (default car)', type=str, choices=['car',],
                               default='car',
                               help='type of object to detect', )
obj_detect_parser.add_argument('time stamp', type=int, help='time stamp of video', default=MIN_TS)
obj_detect_parser.add_argument('x', type=int, help='Reload labels', default=ATTENTION_COOR[1])
obj_detect_parser.add_argument('y', type=int,
                               help='Reload labels', default=ATTENTION_COOR[0])
obj_detect_parser.add_argument('width', type=int, choices=[80], default=80,
                               help='width of input image', )
obj_detect_parser.add_argument('height', type=int, choices=[80], default=80,
                               help='height of input image', )


def init_classifier():
    car_model = build_knn()
    return car_model
global car_model
car_model  = init_classifier()

def init_video_handler():
    import data_utils
    t = data_utils.VideoDatabaseAccess()
    return t
global video_handler
video_handler = init_video_handler()

@ns.route('/ask/contain')
class ObjDetect(Resource):
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
            target_object = payload.get('target_object', 'car')
            time_stamp = payload.get('time stamp', MIN_TS)
            x = payload.get('x', ATTENTION_COOR[1])
            y = payload.get('y', ATTENTION_COOR[0])
            width = payload.get('width', FRAME_SIZE[1])
            height = payload.get('height', FRAME_SIZE[0])
            frame = video_handler.get_frame_given_ts(time_stamp, h=height, w=width, x=x, y=y)
            embedded_frame = encoder.predict(np.array([frame]))
            ret = car_model.predict(embedded_frame)
            this_result = ret[0]
            return make_response(
                jsonify(
                    {
                        'status': 'success!',
                        'result': this_result
                    },
                    200
                )
            )
        except FileNotFoundError as e:
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
    app.run(host='0.0.0.0',port=5000)
    # app.run()