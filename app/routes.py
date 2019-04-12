from flask import render_template, send_from_directory, request, \
   redirect, url_for, flash
from app import app
import os
import sys
sys.path.append("..")
import Main as internalApp

ALLOWED_EXTENSIONS = set(['jpg'])

def allowed_file(filename):
   return '.' in filename and \
      filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    information = {'username':'Chaitanya'}
    return render_template('index.html', title='Home', information=information)

@app.route('/uploadFile', methods=['GET','POST'])
def upload_file():
   if request.method == 'POST':
      # check if the post request has the file part
      if 'file' not in request.files:
         flash('No file part')
         return redirect(request.url)
      file = request.files['file']
      # if user does not select file, browser also submits an empty part without filename
      if file.filename == '':
         flash('No file Selected')
         return redirect(request.url)
      if file and allowed_file(file.filename):
         filename = "0.jpg"
         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
         information = {'result':internalApp.predictType()}
         return render_template('index.html', title='Home', information=information)
   return render_template('uploadFile.html', title='Upload Image')
