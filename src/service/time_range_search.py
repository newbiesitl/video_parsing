from flask import Flask, jsonify, make_response, Blueprint
from flask_restplus import Api, Resource, fields
from model_factory.knn_object_detection import  build_knn
from download_utils import get_index_file





# init app
app = Flask(__name__)
app.config.SWAGGER_UI_DOC_EXPANSION = 'full'
blueprint = Blueprint('api', __name__, url_prefix='/stream_util')
api = Api(blueprint, version='1.0',title='Stream plus', description='Content indexing')
app.register_blueprint(blueprint)
ns = api.namespace('v1', description='Content parsing node')


obj_detect_parser = api.parser()
obj_detect_parser.add_argument('target object (default car)', type=str, choices=['car',], help='type of object to detect', location='form')
obj_detect_parser.add_argument('time stamp', type=int, help='time stamp of video', location='form')
obj_detect_parser.add_argument('x', type=int, help='Reload labels', location='form')
obj_detect_parser.add_argument('y', type=int,
                               help='Reload labels', location='form')
obj_detect_parser.add_argument('width', type=int, choices=[80], help='width of input image', location='form')
obj_detect_parser.add_argument('height', type=int, choices=[80], help='height of input image', location='form')


def init():
    car_model = build_knn()
    return car_model
global car_model
car_model  = init()

@ns.route('/ask/contain')
class ObjDetect(Resource):
    @ns.doc(description='''
    Current detection only support one size image
    ( 80 * 80 )
    ''', parser=obj_detect_parser)
    def get(self):
        '''
        Provide two time stamps and return boolean to indicate if objects in two time stamps are the same
        :return:
        '''
        try:
            payload = obj_detect_parser.parse_args()
            target_object = payload.get('target_object', 'car')
            time_stamp = payload.get('time stamp', 0)
            x = payload.get('x', 0)
            y = payload.get('y', 0)
            width = payload.get('width', 80)
            height = payload.get('height', 80)
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