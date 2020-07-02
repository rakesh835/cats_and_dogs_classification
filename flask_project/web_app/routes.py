from web_app import app

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import os
from keras.preprocessing import image
import uuid
import base64
from flask import send_from_directory
import PIL
from PIL import Image
import numpy as np

#app=Flask(__name__)


model = load_model('cats_vs_dogs_1592820153.h5')

upload_dir="web_app/static/uploaded"
app.config['UPLOAD_FOLDER'] = upload_dir

ALLOWED_EXTENSIONS=set(['png', 'jpg', 'jpeg', 'bmp'])


@app.route('/')
def upload_f():
    return render_template('pred.html', imagesource="/static/g_image.png", ss='')

def finds(path):
    
    vals = ['Cat', 'Dog'] 
    
    image_shape=150
    img=image.load_img(path, target_size=(image_shape, image_shape))
    img_arr=image.img_to_array(img)
    img_to_pred=np.expand_dims(img_arr, axis=0)

    prediction = model.predict(img_to_pred)
    prediction=tf.squeeze(prediction).numpy()
    print("PREDICTION: ",prediction)
    print(vals[int(prediction)])
    return vals[int(prediction)]


def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() 
    random = random.replace("-","") 
    return random[0:string_length] 


def allowed_file(filename):
    return "." in filename and \
    filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploaded', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            val = finds(file_path)

            filename = my_random_string(6) + filename
            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
         
            return render_template('pred.html', ss = val, imagesource="static/uploaded/"+filename)


@app.route('/uploaded/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == '__main__':
    pass
