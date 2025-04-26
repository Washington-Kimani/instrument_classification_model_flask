import tensorflow as tf


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