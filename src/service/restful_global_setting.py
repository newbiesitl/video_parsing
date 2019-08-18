from flask import Flask, Blueprint
from flask_restplus import Api
import data_utils

from model_factory.knn_object_detection import build_knn

app = Flask(__name__)
blueprint = Blueprint('api', __name__, url_prefix='')
api = Api(blueprint, version='1.0',title='Stream plus', description='Content indexing')
ns = api.namespace('v1', description='Content parsing node')

# init app
app.config.SWAGGER_UI_DOC_EXPANSION = 'full'
app.register_blueprint(blueprint)


global car_model
car_model = build_knn()

global model_store
model_store = {}

global video_handler
video_handler = data_utils.VideoDatabaseAccess()







def init_object_detection_app_backend(object_type):
    global model_store
    model_store[object_type] = car_model
