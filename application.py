import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from geoguessr import load_images, FinalModel
import pickle

MODELDIR = './model'

model = FinalModel(MODELDIR)

# saved models

UPLOAD_FOLDER = './static/uploads/'

application = Flask(__name__)
application.secret_key = "secret key"
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
application.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

HR_poly = pickle.load(open("HR_poly.pkl",'rb'))

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_grid(prediction):
    plt.clf()
    for polygon in HR_poly:
        xs, ys = zip(*polygon)
        plt.plot(ys,xs, c='grey')
    plt.scatter(prediction[1], prediction[0], s=200, marker='x', c='r')
    plt.savefig('static/cro_with.png')
    ax = plt.gca()
    ax.axis('off')
    plt.savefig('static/cro.png')
	
@application.route('/')
def upload_form():
	return render_template('upload.html')

@application.route('/', methods=['POST'])
def upload_image():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('files[]')
    file_names = []

    files_to_remove = next(os.walk(application.config['UPLOAD_FOLDER']), (None, None, []))[2]
    for file in files_to_remove:
        os.remove(os.path.join(application.config['UPLOAD_FOLDER'], file))
    

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
        
    product = load_images(application.config['UPLOAD_FOLDER'], file_names)
    prediction = model.predict(product)
    generate_grid(prediction)
    return render_template('upload.html', prediction=prediction, filenames=file_names)

@application.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

 
if __name__ == "__main__":
    application.run(debug=True)