## Behavioral Cloning

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nnvis]: ./examples/ajays_neural_network.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of pre-processing layers - very similar to what was discussed in the lectures, followed by a set of convolutional and pooling layers, and then followed by a few fully connected layers. 

This is the model I converged to after a lot of experimentation. 

![Ajay's neural network for steering angle prediction.][nnvis]

I started off with the LeNet modified model used for traffic signs classification, but I did see that it was inadequate, even after adding a few more fully-connected layers. (More explantion below)

I started reading some literature online about AlexNet and VGG. Both of them in their plain form were too heavyweight for the small number of images I had (on the order of 20K.) The model I used is quite inspired by [this paper](https://arxiv.org/pdf/1604.07316.pdf). Again, I modified the convolutional layers to suit my input image sizes. Also, the sizes of the fully-connected layers are reduced to prevent under-fitting, given my dataset size. 

Salient points about my model. The 5 convolutional layers are used to learn complex shapes (simpler shapes in the first few layers, followed by more complex shapes in later layers). The model needs to output a single floating point number based on the shapes learnt in the convolutional layers. The set of fully connected layers (with the use of dropout) help in this. I initally was using a large number of connections in the fully-connected layers, which resulted in severe under-fitting. This design was a convergence of expermintation of FC layers of various sizes.

#### 2. Attempts to reduce overfitting and under-fitting in the model

1. __Data Augmentation__
  * For each position of the car, we get 3 images from front, left and right cameras. First, the center camera image is shifted by a random direction (up, left, front or right) by 5 pixels. These images are laterally inverted to get a set of 8 images. 
  * Note that I intriduce a slight shift to the center camera angle so that the inverted image gets the negative shift.
  
2. __Data capture from second track__
  * I captured data from the second track to ensure that the model I'm training for consists of images of tracks that we'll not be testing on. This greatly helped in getting past the first big curve. 

3. __Explicit data capture in problematic curves__
  * The track consists of mainly sections where the car drives stright. 
    * So, at curves, the the car is slow to react fast for the first few training epochs.
    * If we use a model that's trained for several epochs, the car drives exactly like the way I do -- it hugs the curves (eerie) --, but slight perturbations can cause the car to go off tracks. So, I drove 10 times around the problematic curves and trained my already trained network. _Thank you Transfer Learning_. See the options in my [model.py](model.py) script to start learning from a saved network.

4. __Data capture from erroneous driving recovery__
  
  * This was a very helpful tip in the lectures. I used the already trained network and captured data only for the part where I was recovering from getting off the tracks.
  * This also made my model slightly prefer corners in driving, but the car for all parts of the course preferred staying close to the edges but not straddle them. 

5. __Reducing the number of parameters of the model__

  * When my training data had consistently high errors, I guessed that I had insufficient data, even after augmentation. I had to reduce the number of parameters in the network (to 239K from over a million). This was one of the first steps that helped me get better with more training. 

#### 3. Model parameter tuning

The model uses Adam optimizer, but my program takes in a parameter for learning rate. I had to do this to ensure that the model does not overtrain when I'm retraining for problematic cases, or for driving around sharp edges. 

#### 4. Appropriate training data

The section on attempts to prevent overfitting discussed how appropriate training data was used. (augmentation, from second track, explicit driving in problematic curves, and recovery from erroneous driving.)

### Model Architecture and Training Strategy

####1. Solution Design Approach

The first step was to ensure I was able to transform the images into a form of trainable data. That's the first version of model.py. I used a linear regression model as suggested in the lecture videos.

The overall strategy for deriving a model architecture was to 

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
