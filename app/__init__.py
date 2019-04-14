from flask import Flask, request, redirect, url_for, flash
from flask_bootstrap import Bootstrap
from config import Config
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './ip'
STATIC_FOLDER = '/static'

app = Flask(__name__, static_url_path=STATIC_FOLDER)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'you-will-never-guess'
from app import routes

bootstrap = Bootstrap(app)

