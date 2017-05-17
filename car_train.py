import csv
import cv2
import numpy as np
import matplotlib as plt
import sklearn
from random import shuffle
from sklearn.model_selection import train_test_split

"""
Helping methods for image transformation
"""
def yuv(img):
    """Convert to YUV space with global and local normalization"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel to image 'img'"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def rotate(center, img, degrees):
    """Rotates imgage 'img' 'degrees'"""
    img_out = cv2.getRotationMatrix2D(center,degrees,1)
    return cv2.warpAffine(img,img_out,(32,32))

def translate(img, pixels):
    """Translate the image 'img' [-2,2] pixels"""
    M = np.float32([[1,0,-pixels],[0,1,pixels]])
    return cv2.warpAffine(img,M,(32,32))

def scale(img, ratio):
    """Scales image 'img' by 'ratio'"""
    return cv2.resize(img, None, fx=ratio, fy=ratio, interpolation = cv2.INTER_CUBIC)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def preProcessImage(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    #image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    #return image

# To store each line of the saved simulator run
# lines = []
# with open('Data/driving_log.csv') as csvfile:
# 	reader = csv.reader(csvfile)
# 	for line in reader:
# 		lines.append(line)
samples = []
with open('./Data/driving_log.csv') as csvfile:
	print("Loading images from Data")
	reader = csv.reader(csvfile)
	for sample in reader:
		samples.append(sample)
	print("Samples Size: {}".format(len(samples)))

with open('./Data_2/driving_log.csv') as csvfile:
	print("Samples Size: {}".format(len(samples)))
	reader = csv.reader(csvfile)
	for sample in reader:
		samples.append(sample)
	print("Samples Size: {}".format(len(samples)))

with open('./Data_track2/driving_log.csv') as csvfile:
	print("Samples Size: {}".format(len(samples)))
	reader = csv.reader(csvfile)
	for sample in reader:
		samples.append(sample)
	print("Samples Size: {}".format(len(samples)))

with open('./Data_smooth/driving_log.csv') as csvfile:
	print("Samples Size: {}".format(len(samples)))
	reader = csv.reader(csvfile)
	for sample in reader:
		samples.append(sample)
	print("Samples Size: {}".format(len(samples)))

# # To save the previously saved images and measurements from simulator
images = []
measurements = []

# def generator(samples, batch_size=32):
# 	num_samples = len(samples)
# 	while 1: # Loop forever so the generator never terminates
# 		shuffle(samples)
# 		for offset in range(0, num_samples, batch_size):
# 			batch_samples = samples[offset:offset+batch_size]

# 			images = []
# 			measurements = []
# 			for batch_sample in batch_samples:
# 				for i in range(3):
# 					source_path = batch_sample[i]
# 					current_path = './Data_generator/IMG/' + source_path.split('/')[-1]
# 					image = cv2.imread(current_path)
# 					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 					# Note:
# 					# All image data from the simulator arrive in RGB
# 					# cv2 imread loads images in BGR
# 					images.append(image)
# 					measurement = float(batch_sample[3])
# 					if i == 0: # center camera
# 						measurements.append(measurement)
# 					elif i == 1: # left camera
# 						measurements.append(measurement + 0.2)
# 					elif i == 2: # right camera
# 						measurements.append(measurement - 0.2)

# 			X_train = np.array(images)
# 			y_train = np.array(measurements)
# 			yield sklearn.utils.shuffle(X_train, y_train)

# Load Data from 1st set of data
for line in samples:	
	# Loading Center Camera Images
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = './Data_generator/IMG/' + filename
		image = cv2.imread(current_path)
		#image = preProcessImage(image)
		images.append(image)
		measurement = float(line[3])
		if i == 0: # center camera
			measurements.append(measurement)
		elif i == 1: # left camera
			measurements.append(measurement + 0.5)
		elif i == 2: # right camera
			measurements.append(measurement - 0.5)
# 
# # Load Data from 2nd set of data
# lines = []
# with open('Data_2/driving_log.csv') as csvfile:
# 	reader = csv.reader(csvfile)
# 	for line in reader:
# 		lines.append(line)
# for line in lines:
# 	# Loading Center Camera Images
# 	for i in range(3):
# 		source_path = line[i]
# 		filename = source_path.split('/')[-1]
# 		current_path = './Data_2/IMG/' + filename
# 		image = cv2.imread(current_path)
# 		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 		images.append(image)
# 		measurement = float(line[3])
# 		if i == 0: # center camera
# 			measurements.append(measurement)
# 		elif i == 1: # left camera
# 			measurements.append(measurement + 0.2)
# 		elif i == 2: # right camera
# 			measurements.append(measurement - 0.2)
# 
# # Load Data from 2nd set of data
# lines = []
# with open('Data_track2/driving_log.csv') as csvfile:
# 	reader = csv.reader(csvfile)
# 	for line in reader:
# 		lines.append(line)
# for line in lines:
# 	# Loading Center Camera Images
# 	for i in range(3):
# 		source_path = line[i]
# 		filename = source_path.split('/')[-1]
# 		current_path = './Data_track2/IMG/' + filename
# 		image = cv2.imread(current_path)
# 		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 		images.append(image)
# 		measurement = float(line[3])
# 		if i == 0: # center camera
# 			measurements.append(measurement)
# 		elif i == 1: # left camera
# 			measurements.append(measurement + 0.2)
# 		elif i == 2: # right camera
# 			measurements.append(measurement - 0.2)
# 
# Data augmentation: flipping images
jitter_images = []
jitter_measurements = []
for image,measurement in zip(images, measurements):
	jitter_images.append(image)
	jitter_measurements.append(measurement)
	jitter_images.append(cv2.flip(image, 1))
	jitter_measurements.append(measurement * -1.0)

for i in range(len(jitter_images)):
	jitter_images[i] = preProcessImage(jitter_images[i])

X_train = np.array(jitter_images)
y_train = np.array(jitter_measurements)
print("X_train Shape: {}".format(X_train.shape))


# Model --------------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2

# Implementing Nvidia Self-driven car CNN model
model = Sequential()

# Cropping first, so any later image manipulation will be cheaper
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160, 320, 3)))

# Normalization layer
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(140,260, 3)))

# YUV layer
#model.add(Lambda(gray, input_shape=(140,260, 3)))

# Convolutional #1
# An 'elu' activation is faster and smoother (lesson learned in P2)
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="elu", W_regularizer=l2(0.001)))
#model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="elu"))
#model.add(MaxPooling2D())

# Convolutional #2
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="elu", W_regularizer=l2(0.001)))
#model.add(MaxPooling2D())

# Convolutional #3
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="elu", W_regularizer=l2(0.001)))
#model.add(MaxPooling2D())

# Convolutional #4
model.add(Convolution2D(64, 3, 3, activation="elu", W_regularizer=l2(0.001)))
#model.add(MaxPooling2D())

# Convolutional #5
model.add(Convolution2D(64, 3, 3, activation="elu", W_regularizer=l2(0.001)))

# Maxpooling
model.add(MaxPooling2D())

# Flatten out layer
model.add(Flatten())

# Flat layer
#model.add(Dense(120))
model.add(Dense(100, activation="elu"))
model.add(Dropout(0.80))

# Flat layer
#model.add(Dense(84))
model.add(Dense(50, activation="elu"))
model.add(Dropout(0.80))

# Flat layer
model.add(Dense(10, activation="elu"))

#Flat layer
model.add(Dense(1))


# Loss/Optimizer operation
#model.compile(loss='mse', optimizer='adam')
model.compile(loss='mse', optimizer=Adam(lr=1e-4))

# --------------------------------------------------------------------------------------

# train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# print("Samples data size: {}".format(len(samples)))
# print("Train data size: {}".format(len(train_samples)))
# print("Validation data size: {}".format(len(validation_samples)))

# # compile and train the model using the generator function
# train_generator = generator(train_samples, batch_size=1)
# validation_generator = generator(validation_samples, batch_size=1)

# Define data preparation
#datagen = ImageDataGenerator(zca_whitening=True)
# Fit parameters from data
#datagen.fit(X_train)
#model.fit_generator(datagen.flow(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=128),
#                    steps_per_epoch=len(X_train) / 128, nb_epoch=10)

# Running the model
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

# model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
# 					validation_data=validation_generator,
# 					nb_val_samples=len(validation_samples), nb_epoch=10)

# Save model
model.save('model.h5')

json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)

# Saving Model visualization
#plot_model(model, to_file='model_visualizaion.png', show_shapes=True)

