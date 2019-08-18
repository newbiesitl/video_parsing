from flask import Flask, Blueprint
from flask_restplus import Api

app = Flask(__name__)
blueprint = Blueprint('api', __name__, url_prefix='')
api = Api(blueprint, version='1.0',title='Stream plus', description='Content indexing')
ns = api.namespace('v1', description='Content parsing node')

# init app
app.config.SWAGGER_UI_DOC_EXPANSION = 'full'
app.register_blueprint(blueprint)

