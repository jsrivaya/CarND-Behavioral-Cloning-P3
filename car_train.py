import csv
import cv2
import numpy as np
import matplotlib as plt

# Suporting methods from preprocessing images
def gray(x):
    return tf.image.rgb_to_grayscale(x)

def yuv(x):
    """Convert to YUV space with global and local normalization"""
    print("X shape: {}".format(x.shape))
    return plt.colors.rgb_to_hsv(x)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

# To store each line of the saved simulator run
lines = []
with open('Data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# To save the previously saved images and measurements from simulator
images = []
measurements = []

# Load Data from 1st set of data
for line in lines:
	# Loading Center Camera Images
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = './Data/IMG/' + filename
		image = cv2.imread(current_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		images.append(image)
		measurement = float(line[3])
		if i == 0: # center camera
			measurements.append(measurement)
		elif i == 1: # left camera
			measurements.append(measurement + 0.2)
		elif i == 2: # right camera
			measurements.append(measurement - 0.2)

# Load Data from 2nd set of data
lines = []
with open('Data_2/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
for line in lines:
	# Loading Center Camera Images
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = './Data_2/IMG/' + filename
		image = cv2.imread(current_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		images.append(image)
		measurement = float(line[3])
		if i == 0: # center camera
			measurements.append(measurement)
		elif i == 1: # left camera
			measurements.append(measurement + 0.2)
		elif i == 2: # right camera
			measurements.append(measurement - 0.2)

# Load Data from 2nd set of data
lines = []
with open('Data_track2/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
for line in lines:
	# Loading Center Camera Images
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = './Data_track2/IMG/' + filename
		image = cv2.imread(current_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		images.append(image)
		measurement = float(line[3])
		if i == 0: # center camera
			measurements.append(measurement)
		elif i == 1: # left camera
			measurements.append(measurement + 0.2)
		elif i == 2: # right camera
			measurements.append(measurement - 0.2)

# Data augmentation: flipping images
jitter_images = []
jitter_measurements = []
for image,measurement in zip(images, measurements):
	jitter_images.append(image)
	jitter_measurements.append(measurement)
	jitter_images.append(cv2.flip(image, 1))
	jitter_measurements.append(measurement * -1.0)

X_train = np.array(jitter_images)
y_train = np.array(jitter_measurements)
print("X_train Shape: {}".format(X_train.shape))


# Model --------------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
#from keras.utils import plot_model

# Note:
# All image data from the simulator arrive in RGB
# cv2 imread loads images in BGR

# Implementing Nvidia Self-driven car CNN model
model = Sequential()

# Cropping first, so any later image manipulation will be cheaper
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160, 320, 3)))

# Normalization layer
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(140,260, 3)))

# Grayscale layer
#model.add(Lambda(gray, input_shape=(140,260, 3)))

# Convolutional #1
# An 'elu' activation is faster and smoother (lesson learned in P2)
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="elu"))
#model.add(MaxPooling2D())

# Convolutional #2
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="elu"))
#model.add(MaxPooling2D())

# Convolutional #3
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="elu"))
#model.add(MaxPooling2D())

# Convolutional #4
model.add(Convolution2D(64, 3, 3, activation="elu"))
#model.add(MaxPooling2D())

# Convolutional #5
model.add(Convolution2D(64, 3, 3, activation="elu"))

# Maxpooling
model.add(MaxPooling2D())

# Flatten out layer
model.add(Flatten())

# Flat layer
model.add(Dense(120))
#model.add(Dropout(0.80))

# Flat layer
model.add(Dense(84))
#model.add(Dropout(0.80))

#Flat layer
model.add(Dense(1))

# --------------------------------------------------------------------------------------

# Loss/Optimizer operation
model.compile(loss='mse', optimizer='adam')

# Define data preparation
#datagen = ImageDataGenerator(zca_whitening=True)
# Fit parameters from data
#datagen.fit(X_train)

#model.fit_generator(datagen.flow(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=128),
#                    steps_per_epoch=len(X_train) / 128, nb_epoch=10)

# Running the model
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

# Save model
model.save('model.h5')

# Saving Model visualization
#plot_model(model, to_file='model_visualizaion.png', show_shapes=True)

