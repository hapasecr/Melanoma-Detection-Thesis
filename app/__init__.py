from flask import Flask, request, redirect, url_for, flash
from flask_bootstrap import Bootstrap
from config import Config
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './ip'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'you-will-never-guess'
from app import routes

bootstrap = Bootstrap(app)