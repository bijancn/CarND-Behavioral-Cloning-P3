# Behavioral Cloning

[//]: # (Image References)

[model]: ./pictures/model.png "Model Visualization"
[NVIDIA blog post]: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
[broken_dist]: ./pictures/broken_distribution.png
[good_dist]: ./pictures/corrected_distribution.png

## Changes after review

After the review, I had to rework major aspects of this project to get
it to finally work. I think the first version was just accidentally
almost good enough. Some of the experiments are documented in a horrible
format in the [lab journal](lab_journal.md). I try to summarize the
changes here:

- I made use of keras `fit_generator`. 

- At some point I change the minimal angle when to consider an image to
  `0.0001` and the left/right correction factor to `0.10` but I don't think
  it makes a big difference.

I adapted the report below to correspond the latest version.

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
learn. Thus, I removed all images with a steering angle `< 0.0001`, which
was about half of the sample. This gave a lot more sensible distribution
of steering angles:

![alt text][good_dist]

### Final Model Architecture

The final model architecture is as described above. Here is a
visualization of the architecture

![alt text][model]

The final training looks like this
```
Train on 59800 samples, validate on ? samples
Epoch 2/15
59600/59800 [============================>.] - val_loss improved from 0.07234 to 0.06671, saving model to weights.01-0.07.hdf5
59800/59800 [==============================] - 119s - loss: 0.0807 - val_loss: 0.0667
Epoch 3/15
59600/59800 [============================>.] - val_loss improved from 0.06671 to 0.06267, saving model to weights.02-0.06.hdf5
59800/59800 [==============================] - 120s - loss: 0.0770 - val_loss: 0.0627
Epoch 4/15
59600/59800 [============================>.] - val_loss improved from 0.06267 to 0.06068, saving model to weights.03-0.06.hdf5
59800/59800 [==============================] - 119s - loss: 0.0746 - val_loss: 0.0607
Epoch 5/15
59600/59800 [============================>.] - val_loss improved from 0.06068 to 0.05939, saving model to weights.04-0.06.hdf5
59800/59800 [==============================] - 119s - loss: 0.0728 - val_loss: 0.0594
Epoch 6/15
59600/59800 [============================>.] - val_loss improved from 0.05939 to 0.05847, saving model to weights.05-0.06.hdf5
59800/59800 [==============================] - 120s - loss: 0.0714 - val_loss: 0.0585
Epoch 7/15
59600/59800 [============================>.] - val_loss improved from 0.05847 to 0.05817, saving model to weights.06-0.06.hdf5
59800/59800 [==============================] - 119s - loss: 0.0704 - val_loss: 0.0582
Epoch 8/15
59600/59800 [============================>.] - val_loss improved from 0.05817 to 0.05789, saving model to weights.07-0.06.hdf5
59800/59800 [==============================] - 119s - loss: 0.0698 - val_loss: 0.0579
Epoch 9/15
59600/59800 [============================>.] - val_loss improved from 0.05789 to 0.05766, saving model to weights.08-0.06.hdf5
59800/59800 [==============================] - 119s - loss: 0.0692 - val_loss: 0.0577
Epoch 10/15
59600/59800 [============================>.] - val_loss improved from 0.05766 to 0.05721, saving model to weights.09-0.06.hdf5
59800/59800 [==============================] - 119s - loss: 0.0686 - val_loss: 0.0572
Epoch 11/15
59600/59800 [============================>.] - val_loss improved from 0.05721 to 0.05706, saving model to weights.10-0.06.hdf5
59800/59800 [==============================] - 119s - loss: 0.0675 - val_loss: 0.0571
Epoch 12/15
59600/59800 [============================>.] - val_loss improved from 0.05706 to 0.05665, saving model to weights.11-0.06.hdf5
59800/59800 [==============================] - 119s - loss: 0.0667 - val_loss: 0.0566
Epoch 13/15
59600/59800 [============================>.] - val_loss improved from 0.05665 to 0.05661, saving model to weights.12-0.06.hdf5
59800/59800 [==============================] - 119s - loss: 0.0657 - val_loss: 0.0566
Epoch 14/15
59600/59800 [============================>.] - val_loss did not improve
59800/59800 [==============================] - 119s - loss: 0.0651 - val_loss: 0.0568
Epoch 15/15
59600/59800 [============================>.] - val_loss did not improve
59800/59800 [==============================] - 119s - loss: 0.0642 - val_loss: 0.0576
```

