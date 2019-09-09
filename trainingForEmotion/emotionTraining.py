import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout, Input, Flatten
from keras.models import Model
from keras.models import Sequential
from keras import optimizers
from keras.optimizers import Adam
from keras.optimizers import Adamax
from keras.optimizers import SGD
from keras.optimizers import rmsprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import *

from sklearn.model_selection import *

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter


def pandas_vector_to_list(pandas_df):
    py_list = [item[0] for item in pandas_df.values.tolist()]
    return py_list
def process_pixels(pixels, img_size=48):
    """
    Takes in a string (pixels) that has space separated ints. Will transform the ints
    to a 48x48 matrix of floats(/255).
    :param pixels: string with space separated ints
    :param img_size: image size
    :return: array of 48x48 matrices
    """
    pixels_as_list = pandas_vector_to_list(pixels)

    np_image_array = []
    for index, item in enumerate(pixels_as_list):
        # 48x48
        data = np.zeros((img_size, img_size), dtype=np.uint8)
        # split space separated ints
        pixel_data = item.split()

        # 0 -> 47, loop through the rows
        for i in range(0, img_size):
            # (0 = 0), (1 = 47), (2 = 94), ...
            pixel_index = i * img_size
            # (0 = [0:47]), (1 = [47: 94]), (2 = [94, 141]), ...
            data[i] = pixel_data[pixel_index:pixel_index + img_size]

        np_image_array.append(np.array(data))

    np_image_array = np.array(np_image_array)
    # convert to float and divide by 255
    np_image_array = np_image_array.astype('float32') / 255.0
    print("\nThe pixel shpe is: \n")
    print(np_image_array.shape);
    return np_image_array

#Main program starts from here
f=pd.read_csv('path_to_file')#Put the correct pathname here
print(f.info);

x_train, y_train, x_test, y_test = [], [], [], []

pixels = process_pixels(f[['pixels']])

emotion_as_list = pandas_vector_to_list(f[['emotion']])

y_data = []
for index in range(len(emotion_as_list)):
      y_data.append(emotion_as_list[index])

    # Y data
y_data_categorical = np_utils.to_categorical(y_data,7)
   
emotion = y_data_categorical
print("Done till here hey");
'''tree=f.iloc[:, -1]
count=0;

for i in range(1, 35887):
 if 'Training' in tree[i]:
  y_train.append(emotion[i])
  x_train.append(pixels[i])
 elif 'PublicTest' in tree[i]:
  y_test.append(emotion[i])
  x_test.append(pixels[i])'''
  
Y=np.array(emotion)
X=np.array(pixels)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#print("\nx_train: " + x_train);

print("Model begins here: \n")
#1st convolution layer
model = Sequential()
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
 
#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
 
#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model.add(Flatten())
 
#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
 
model.add(Dense(7, activation='softmax'))

print(model.summary())


x_train=np.array(x_train)
y_train=np.array(y_train)
x_train=x_train.reshape(x_train.shape[0],48,48,1)

x_test=np.array(x_test)
y_test=np.array(y_test)
print("xtest is here\n") 
print(x_test.shape);

gen = ImageDataGenerator(featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

train_generator = gen.flow(x_train, y_train, batch_size=1024)


'''model.compile(loss='categorical_crossentropy'
, optimizer=opt
, metric=['accuracy']
)'''

#lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
#early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
obj=Adamax()
model.compile(optimizer=obj,
              loss='categorical_crossentropy',
              metrics=['accuracy']) 
print("Entering training era")

model.fit_generator(train_generator, steps_per_epoch=1024, epochs=50)

train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1])
 

x_test=x_test.reshape(x_test.shape[0], 48,48,1)

test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', 100*test_score[1])

predictions = model.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
 
pred_list = []; actual_list = []
 
for i in predictions:
 
   pred_list.append(np.argmax(i))
 
for i in y_test:
 
   actual_list.append(np.argmax(i))
 
print(confusion_matrix(actual_list, pred_list))

from keras.models import load_model

#model = load_model('myMoodModel.h5')
#print("\nNew model is here\n")
#print(model.summary())
model_json = model.to_json()
with open("path/myMoodModel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("path/model_weights.h5")
print("Saved model to drive")
#Save the model to google drive from colab