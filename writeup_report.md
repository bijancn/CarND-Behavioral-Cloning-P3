# **Behavioral Cloning**

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[NVIDIA blog post]: https://devblogs.nvidia.com/deep-learning-self-driving-cars/

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup_report.md` or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can
be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the
convolution neural network. The file shows the pipeline I used for
training and validating the model, and it contains comments to explain
how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The keras model is created in `setup_model`. It follows the recommended
architecture of the [NVIDIA blog post].
The data is normalized in the model using a Keras lambda layer and
cropped at the top and bottom.
The neural network itself consists of five convolutional layers, each
followed by a nonlinear RELU function, and four fully connected layers.
The first three layers have a 2x2 stride with a 5x5 kernel followed by
two layers with a 3x3 kernel without stride.
These configurations have been found empirically by the NVIDIA team.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure
that the model was not overfitting (code line 10-16). The model was
tested by running it through the simulator and ensuring that the vehicle
could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned
manually. As an error measure, I chose `mean_absolute_error` to
encourage the network to take a turn as errors are less heavily
penalized than with `mean_squared_error`.

#### 4. Appropriate training data

I have started with the central images of the supplied test data
(`data`). This got me until the dirt corner where the model preferred
the offroad path. Thus I added some samples around this corner
(`dirtcorner`). This threw it off-balance though, so I added some more
content with `more` and `dirt_extra`.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I was fairly confident in the NVIDIA architecture as it is a very simple
set of layers that was already tested on real streets.

To monitor overfitting, I chose a validation split of 0.2. This showed
that while the training loss did still improve slightly with more
epochs, validation loss was saturated quickly. To have quicker
iterations and avoid overfitting, I thus only chose 3 epochs.

The final step was to run the simulator to see how well the car was
driving around track one. There were a few spots where the vehicle fell
off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously
around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a
convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the
architecture is optional according to the project rubric)

![alt text][image1]

The final training looks like this
```
Train on 14668 samples, validate on 3668 samples
Epoch 1/3
14668/14668 [==============================] - 80s 5ms/step - loss: 0.0858 - val_loss: 0.0690
Epoch 2/3
14668/14668 [==============================] - 85s 6ms/step - loss: 0.0754 - val_loss: 0.0774
Epoch 3/3
14668/14668 [==============================] - 84s 6ms/step - loss: 0.0710 - val_loss: 0.0637
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I started with the test data and then
added my behavior in the corners where it was necessary. Here is an
example image of center lane driving:

![alt text][image2]

To add recovery scenarios without recording to many of my likely flawed
behavior, I added the side camera images with a correction factor of
`0.10`. The factor was found experimentally by checking the driving
behavior. I also checked the distribution of steering angles, c.f.
`analyze_angles`, and found t

![alt text][image1]

After the collection process, I had X number of data points. I then
preprocessed this data by ...

I finally randomly shuffled the data set and put Y% of the data into a
validation set.

I used this training data for training the model. The validation set
helped determine if the model was over or under fitting. The ideal
number of epochs was Z as evidenced by ... I used an adam optimizer so
that manually training the learning rate wasn't necessary.
