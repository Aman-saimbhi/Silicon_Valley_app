import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as k
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from flask import request
from flask import jsonify
from flask import Flask
from keras.optimizers import Adam
from keras.applications import VGG16
from keras.applications.vgg16 import VGG16
from keras.backend import set_session
import tensorflow as tf



app = Flask(__name__)

def get_model():

     with graph.as_default():
         set_session(sess)
         global model
         vgg_16_model=VGG16()
         model=Sequential()
         for layer in vgg_16_model.layers[:-1]:
             model.add(layer)
         for layer in model.layers:
             layer.trainable=False
         model.add(Dense(1, activation='sigmoid'))
         model.load_weights('silicon_valley_model.h5')

         model._make_predict_function()

         print(" * Model loaded!")




def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array (image)
    image = np.expand_dims(image, axis=0)

    return image



print(' * Loading Keras model.. ')
global graph
global sess
sess=tf.Session()
graph = tf.get_default_graph()
get_model()

@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    with graph.as_default():
        set_session(sess)
        prediction = model.predict(processed_image)
        val=float(np.squeeze(prediction[0][0]))
        #print(1-val)
        response = {
             'prediction' : {
                  'hot_dog': val,
                  'not_hot_dog': 1 - val
                        }
                }

    return jsonify(response)
