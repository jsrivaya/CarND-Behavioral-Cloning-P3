
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/NvidiaArchitecture.png "Model Visualization"
[image2]: ./writeup_images/training_epochs.png "Epochs"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
* I have implemented Nvidia Self-Driven Car CNN Architeture in this project. This consist in 5 Convolutional Layers and 3 Flat Layers. I have added on top of that a Poolmax Layer. Also, for the activation I have used an 'ELU' function instad of 'RELU', it is faster. Normalization is included as a first layer as part of the CNN. This makes it faster process. 
* The collected data has been devided in a 80/20 data sets for training and validation. Also, I have included Dropout layers in the between the first two flat layers to avoid overfitting.
* I have decided to use an Adam optimizer since it provided good results in the traffic sign recognition lab.
* The training data was collected in various phases.
  * One normal lap
  * One uncompleted lap going out and back in the track
  * One lap in the second track. This turned out to be a game changer and stated having much better results. The reson looks to be that the second track includes more winding roads than the first track so the model learns how to do sharper turns
  * One smooth lap in the first track.
  * All Data is flipped to simulate mirror driving
  * All Data is preprocessed with a Gaussian Blur filter
  * All Data is preprocessed to use YUV color space. The reason for this is that the CNN perform better in this kind of color spaces.

---
###Files Submitted & Code Quality

My project includes the following files:
* car_train.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The car_train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes ELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... I used also al cameras available for the training. As suggested in the lesson using left and right camera to apply an extra turn correction.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I use regularizer in all the layers.

Then I apply Dropout between the first two flat layers with a retention rate of 80%

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (car_train.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture

![alt text][image1]

####3. Creation of the Training Set & Training Process

The training data was collected in various phases.
  * One normal lap in target track
  * One uncompleted lap going out of the track and recovering. This was done for both right and left sides of the road
  * One lap in the second track. This turned out to be a game changer and stated having much better results. The reson looks to be that the second track includes more winding roads than the first track so the model learns how to do sharper turns
  * One smooth lap in the target track.
  * All Data is flipped to simulate mirror driving

After the collection process, I had 43878 number of data points. I then preprocessed this data by:
* Using Gaussian Blur filter with a 3x3 kernel
* Convert to YUV color space. The reason for this is that the CNN perform better in this kind of color spaces.
* Normalization. This is done as a CNN layer
* Cropped to simplify the learning process for the CNN.

Bellow is an example of input image and what the CNN sees

![alt text][image2]

![alt text][image3]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the fact that both training loss and validation loss decrease accordingly and in a stady way. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image2]
