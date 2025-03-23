from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# load the model
model = load_model(os.path.join('models', 'instrument_classification.h5'))

# the array of instrument categories
data_cat = ['Didgeridoo',
 'Tambourine',
 'Xylophone',
 'acordian',
 'alphorn',
 'bagpipes',
 'banjo',
 'bongo drum',
 'casaba',
 'castanets',
 'clarinet',
 'clavichord',
 'concertina',
 'drums',
 'dulcimer',
 'flute',
 'guiro',
 'guitar',
 'harmonica',
 'harp',
 'marakas',
 'ocarina',
 'piano',
 'saxaphone',
 'sitar',
 'steel drum',
 'trombone',
 'trumpet',
 'tuba',
 'violin']

# Image dimensions
img_height = 224
img_width = 224

# prediction function
def predict_label(image_path):
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))

    # Preprocess the image
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    # Make prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    return data_cat[np.argmax(score)]


# home route render the index page
@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('index.html')

# about page
@app.route('/about')
def about_page():
    return render_template('about.html', title="About Page")

# prediction route for making the prediction
@app.route('/submit', methods=['GET', 'POST'])
def get_prediction():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = 'static/' + img.filename

        # save image to static folder
        img.save(img_path)

        #call the prediction method
        p = predict_label(img_path)

        return render_template("index.html", prediction = p, img_path = img_path)

if __name__ == '__main__':
    app.run(debug=True)