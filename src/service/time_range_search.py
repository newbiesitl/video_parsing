from flask import Flask, jsonify, make_response, Blueprint
from flask_restplus import Api, Resource, fields





# init app
app = Flask(__name__)
app.config.SWAGGER_UI_DOC_EXPANSION = 'full'
blueprint = Blueprint('api', __name__, url_prefix='/ask')
api = Api(blueprint, version='1.0',title='Stream plus', description='Content indexing')
app.register_blueprint(blueprint)
ns = api.namespace('v1', description='Parking lock')


detect_car_parser = api.parser()
detect_car_parser.add_argument('algorithm', type=str, choices=['topic_prediction', 'popularity_prediction'], help='Reload labels', location='form')
detect_car_parser.add_argument('normalized hot news threshold', type=float, choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], help='Reload labels', location='form')
detect_car_parser.add_argument('normalized breaking news threshold', type=float, choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], help='Reload labels', location='form')
