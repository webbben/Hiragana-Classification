
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#utility functions for reading/cleaning data from ETL
from etl_reader import *

#Files I get images from
#ETL4: dataset of 6113 images of hiragana characters
#ETL7: dataset of 9200 images of hiragana characters (has more images, but the rest aren't hiragana)
#The number associated is the ETL type, so use it to point toward how to read file in functions
files = [('ETL4/ETL4C', 4), ('ETL7/ETL7LC_1', 1), ('ETL7/ETL7SC_1', 1)]

labels = [] #collects labels corresponding to each image
#Get image data and labels from each file in the files list
for i in range(len(files)):
	filename = files[i][0]
	etl = files[i][1]
	new_data = create_data(filename, etl=etl)
	labels1 = getLabelInfo(filename, etl=etl)[0]
	labels = np.concatenate([labels, labels1]) #adds files labels to labels list
	if i == 0:
		print("shape of data before setting:", new_data.shape)
		data = new_data.copy() #overwrite
		print("initialized data shape:", data.shape)
	else:
		print(data.shape)
		print(new_data.shape)
		data = np.concatenate([data, new_data])


data = np.array(data, dtype = np.uint8) # 8 bit unsigned pictures for opencv
data1 = data.copy() #copy by value(and not by reference)
data1 = preprocessing_data(data1, data) #this step applies gaussian blur to remove noise from images

print("number of images:", len(data1))
print("number of labels:", len(labels))


#plt.figure()
#plt.imshow(data1[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()
#plt.show()

#scale the data for the nn to values 0 to 1 range
data1 = scaleData(data1)

#class types for the data (types of hiragana characters, excluding obsolete ones like Yi, Ye, Wi, We etc)
classes = ['A', 'I', 'U', 'E', 'O', 'KA', 'KI', 'KU', 'KE', 'KO', 'SA', 'SI', 'SU', 'SE', 'SO', 'TA', 'TI', 'TU', 'TE', 'TO', 'NA', 'NI', 'NU', 'NE', 'NO', 'HA', 'HI', 'HU', 'HE', 'HO', 'MA', 'MI', 'MU', 'ME', 'MO', 'YA', 'YU', 'YO', 'RA', 'RI', 'RU', 'RE', 'RO', 'WA', 'WO', 'N']

classes_dict = {} #generates and stores numerical version of classes
for i in range(len(classes)):
    classes_dict[classes[i]] = i

#turns labels into their corresponding number
for i in range(len(labels)):
    labels[i] = classes_dict[labels[i]]

#shuffle the images (and keep them linked to their labels)
#do this by making (image, label) tuples and shuffling them
shuffled_images, shuffled_labels = shuffleData(data1, labels)

#training and testing sets

if (len(data1) != len(labels)):
	print("Error: data and labels not same size!")

split = round(len(data1) * (3/4)) #splits data 3/4 training 1/4 testing

print("splitting data Train/Test at index " + str(split))

train_images = shuffled_images[:split]
train_labels = shuffled_labels[:split]
test_images = shuffled_images[split:]
test_labels = shuffled_labels[split:]

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


#======================
# Neural Network Model
#======================

model = keras.Sequential([
    keras.layers.Flatten(input_shape=data1[0].shape),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

model.save('hiragana_classifier.h5') #saves the model to an hdf5 file

