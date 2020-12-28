# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:28:35 2020

@author: Altug
"""

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from scipy import stats
import c_writer
from keras.utils.vis_utils import plot_model


# Print versions
print('Numpy ' + np.__version__)
print('TensorFlow ' + tf.__version__)
print('Keras ' + tf.keras.__version__)


# Settings
models_path = 'C:/Users/Altug/Desktop/Ai_in_Mcu/'  # Where we can find the model files (relative path location)
keras_model_name = 'fisher'           # Will be given .h5 suffix
tflite_model_name = 'fisher'          # Will be given .tflite suffix

# Load model
print(join(models_path, keras_model_name) + '.h5')
model = models.load_model('fisher.h5')

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Convert Keras model to a tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(tflite_model_name+ '.tflite', 'wb').write(tflite_model)

