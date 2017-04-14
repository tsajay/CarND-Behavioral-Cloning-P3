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
[video]: ./video.mp4 "Video of the car driving autonomously."
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

#### 1. Solution Design Approach

It would be a sham to say that I started off with a solution design approach. This was a learning experience, with lots of experimentation, for me, as well as for my model.

The small network used to classify numbers or traffic signals proved inadequeate for staying on the course even for the first 2 curves. 

I then researched the well known networks like AlexNet, VGG and GoogLeNet. They're all too big (too many parameters) for a training set of my size (~80K training + validation images after augmentation). 

The data augmentation section also discusses some of the approaches that I tried and failed, before I came up with this network.


#### 2. Final Model Architecture

The network I arrived at is inspired from [this paper] (https://arxiv.org/pdf/1604.07316.pdf). The parameters are pruned to fit my image size. The number of parameters in the fully-connected layer are pruned to accommodate for my training set size.

Training this network on my image size results in the following characteristics.
* The first few epochs of training generalize well, and the steering angles are scaled slightly less than ideal at curves since most of the course is straight. Still, the direction of steering is correct. One can use the training from initial epochs and add a factor in the drive.py to drive within the course from the first few epochs of training itself. __This is my chosen approach__
* The last few epochs of training tend to overfit in terms of driving too close to the lane markings. Though the car still stays within course for the most part, minor perturbations are enough to steer it off course. 

Hence, though I have several networks where the car can complete laps from later-epoch-trained networks (> 10), I chose to submit the more general network where adjusting the steering angle with a correction factor is sufficient for the car to go around the course indefinitely.  

The sections above show the network architecture. 

#### 3. Creation of the Training Set & Training Process

This is covered in reduce overfitting section.

Here's an image from the second track.

![alt text][image2]

Here's an image from an attempted recovery from driving off course. 

Here are a few augmented images. 
![alt text][image3]
![alt text][image4]
![alt text][image5]


After the collection process, I had 69K training images and close to 90K total images, 20% of which were used for validation. 

Here's a video of the car driving autonomously for more than a lap.

![Video of the car driving autonomously][video.mp4]