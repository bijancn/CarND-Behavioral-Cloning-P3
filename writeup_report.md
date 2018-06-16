# Behavioral Cloning

[//]: # (Image References)

[model]: ./pictures/model.png "Model Visualization"
[NVIDIA blog post]: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
[broken_dist]: ./pictures/broken_distribution.png
[good_dist]: ./pictures/corrected_distribution.png

## Files Submitted & Code Quality

### 1. Submission includes all required files

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup_report.md` or writeup_report.pdf summarizing the results
* [Video of this model driving](https://www.youtube.com/watch?v=oHLq4uueTrw)

### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can
be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
(I doubled the speed as it made no difference in when the car goes
offtrack)

### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the
convolution neural network. The file shows the pipeline I used for
training and validating the model, and it contains comments to explain
how the code works.

## Model Architecture and Training Strategy

### An appropriate model architecture has been employed

The keras model is created in `setup_model`. It follows the recommended
architecture of the [NVIDIA blog post].
The data is normalized in the model using a Keras lambda layer and
cropped at the top and bottom.
The neural network itself consists of five convolutional layers, each
followed by a nonlinear RELU function, and four fully connected layers.
The first three layers have a 2x2 stride with a 5x5 kernel followed by
two layers with a 3x3 kernel without stride.
These configurations have been found empirically by the NVIDIA team.

### Attempts to reduce overfitting in the model

The model was trained and validated on four different data sets to ensure
that the model was not overfitting. The model was tested by running it
through the simulator and ensuring that the vehicle could stay on the
track.

### Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned
manually. As an error measure, I chose `mean_absolute_error` to
encourage the network to take a turn as errors are less heavily
penalized than with `mean_squared_error`.

### Creation of appropriate training data & training process

To capture good driving behavior, I started with the test data (`data`)
and then added my behavior in the corners where it was necessary.
The test data got me until the dirt corner where the model preferred
the offroad path. Thus I added some samples around this corner
(`dirtcorner`). This threw it off-balance though, so I added some more
content with `more` and `dirt_extra`.

To add recovery scenarios without recording to many of my likely flawed
behavior, I added the side camera images with a correction factor of
`0.10`. The factor was found experimentally by checking the driving
behavior.

After initial collection process, I had ~30.000 number of data points.
The preprocessing has been done in the first layer of the model as
described above. I randomly shuffled the data set and put 20% of the
data into a validation set.

### Solution Design Approach

I was fairly confident in the NVIDIA architecture as it is a simple
set of layers that was already tested on real streets.

To monitor overfitting, I chose a validation split of 0.2. This showed
that while the training loss did still improve slightly with more
epochs, validation loss was saturated quickly. To have quicker
iterations and avoid overfitting, I thus only chose 3 epochs.

One major finding in analyzing problems with the model was that my test
set was dominated by *lazy straight ahead driving*:

![alt text][broken_dist]

This was however the least important piece of data I wanted the model to
learn. Thus, I removed all images with a steering angle `< 0.001`, which
was about half of the sample. This gave a lot more sensible distribution
of steering angles:

![alt text][good_dist]

### Final Model Architecture

The final model architecture is as described above. Here is a
visualization of the architecture

![alt text][model]

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
