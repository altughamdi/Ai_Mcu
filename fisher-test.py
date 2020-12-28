# lstm autoencoder recreate sequence
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from sklearn import preprocessing
from os import listdir
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.utils import shuffle

folder_path='C:/Users/Altug/Desktop/Ai_in_Mcu/'
dataset_path =  join(folder_path+'dataset') # Directory where raw accelerometer data is stored
normal_op_list = ['specs'] #['fan_0_low_0_weight']

val_ratio = 0.2             # Percentage of samples that should be held for validation set
test_ratio = 0.2            # Percentage of samples that should be held for test set

keras_model_name = join(folder_path+'fisheriris')        # Will be given .h5 suffix

    
specs = []
sample = np.genfromtxt("dataset/specs/0001.csv", delimiter=',')
specs=sample
specs=specs[~np.isnan(specs).any(axis=1)]

result = []
res = np.genfromtxt("dataset/res/0002.csv", delimiter=',')
result=res

specs, result = shuffle(specs, result)

m_in=len(specs)
n_in=len(specs[0])

val_set_size = m_in * val_ratio
test_set_size = m_in * test_ratio



x_val = specs[:round(val_set_size)]
x_test = specs[round(val_set_size):round(val_set_size + test_set_size)]
x_train = specs[round(val_set_size + test_set_size):]

res_val = result[:round(val_set_size)]
res_test = result[round(val_set_size):round(val_set_size + test_set_size)]
res_train = result[round(val_set_size + test_set_size):]

      # Number of nodes in first layer
model = models.Sequential([
    layers.InputLayer(input_shape=n_in),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(8, activation='relu'), 
    layers.Dropout(0.2),
    layers.Dense(4, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(2, activation='relu'),
    layers.Dense(1, activation='relu'),

])

# Display model
model.summary()

# Add training parameters to model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
             loss='mse')

# Train model (note Y labels are same as inputs, X)
history = model.fit(x_train,
                   res_train,
                   epochs=400,
                   batch_size=100,
                   validation_data=(x_val, res_val),
                   verbose=1)

#%tensorboard --logdir logs

# Plot results
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


yhat = model.predict(x_test, verbose=0)
# model.predict([[6.3, 2.7, 4.9, 1.8]])

model.save( 'fisher.h5')
weights1 = model.get_weights() 