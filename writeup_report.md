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


### 2018-06-30 12:54
Loss doesnt go below 0.07. It looks like more epochs could help
Time:
real    22m47.518s
user    14m3.780s
sys     5m21.340s
Epoch 1693/1695                                                                                                                                                                                                                   200/200 [==============================] - 0s - loss: 0.1971                                                                                                                                                                      Epoch 1694/1695                                                                                                                                                                                                                   200/200 [==============================] - 0s - loss: 0.1032                                                                                                                                                                      Epoch 1695/1695                                                                                                                                                                                                                   200/200 [==============================] - 0s - loss: 0.2520

Model drives almost straight and thus goes out on the right side

### 2018-06-30 12:54
#### Changes
- Increased epochs from 5 to 7
- samples_per_epoch=sample_size and nb_epochs=7
#### Observations
- Now we have about 4 minutes per epoch and more constant loss result

Epoch 1/7
67800/67800 [==============================] - 262s - loss: 0.2152
Epoch 2/7
67800/67800 [==============================] - 258s - loss: 0.1921
Epoch 3/7
67800/67800 [==============================] - 258s - loss: 0.1811
Epoch 4/7
67800/67800 [==============================] - 258s - loss: 0.1761
Epoch 5/7
67800/67800 [==============================] - 258s - loss: 0.1882
Epoch 6/7
67800/67800 [==============================] - 258s - loss: 0.1760
Epoch 7/7
67800/67800 [==============================] - 258s - loss: 0.1665

So it still improves even in the 7th epoch

Model driving behavior has improved somewhat but still makes a pretty
dumb impression
#### Conclusions
- So probably no performance change from the reorganization
- More epochs could still help

### 2018-06-30 13:31
#### Changes
- Increased epochs from 7 to 10
- Removed straight driving filter and left and right images
#### Observations
Epoch 1/10
37200/37200 [==============================] - 149s - loss: 0.1910
Epoch 2/10
37200/37200 [==============================] - 144s - loss: 0.1327
Epoch 3/10
37200/37200 [==============================] - 144s - loss: 0.1253
Epoch 4/10
37200/37200 [==============================] - 144s - loss: 0.1267
Epoch 5/10
37200/37200 [==============================] - 144s - loss: 0.1203
Epoch 6/10
37200/37200 [==============================] - 144s - loss: 0.1162
Epoch 7/10
37200/37200 [==============================] - 144s - loss: 0.1264
Epoch 8/10
37200/37200 [==============================] - 144s - loss: 0.1220
Epoch 9/10
37200/37200 [==============================] - 144s - loss: 0.1189
Epoch 10/10
37200/37200 [==============================] - 144s - loss: 0.1188

Model actively stears off to the right side.
#### Conclusions
- While the loss itself is lower the behavior looked straight up wrong

### 2018-06-30 14:46
#### Changes
- Increased epochs from 10 to 20
- Readded left and right images (kept straight driving filter out)
#### Observations
Epoch 1/20
111600/111600 [==============================] - 437s - loss: 0.1716
Epoch 2/20
111600/111600 [==============================] - 436s - loss: 0.1527
Epoch 3/20
111600/111600 [==============================] - 438s - loss: 0.1435
Epoch 4/20
111600/111600 [==============================] - 442s - loss: 0.1529
Epoch 5/20
111600/111600 [==============================] - 435s - loss: 0.1394
Epoch 6/20
111600/111600 [==============================] - 446s - loss: 0.1498
Epoch 7/20
111600/111600 [==============================] - 443s - loss: 0.1495
Epoch 8/20
111600/111600 [==============================] - 436s - loss: 0.1489
Epoch 9/20
111600/111600 [==============================] - 433s - loss: 0.1421
Epoch 10/20
111600/111600 [==============================] - 433s - loss: 0.1405
Epoch 11/20
111600/111600 [==============================] - 436s - loss: 0.1502
Epoch 12/20
111600/111600 [==============================] - 448s - loss: 0.1500
Epoch 13/20
111600/111600 [==============================] - 458s - loss: 0.1539
Epoch 14/20
111600/111600 [==============================] - 447s - loss: 0.1528
Epoch 15/20
111600/111600 [==============================] - 460s - loss: 0.1512
Epoch 16/20
111600/111600 [==============================] - 477s - loss: 0.1371
Epoch 17/20
111600/111600 [==============================] - 478s - loss: 0.1243
Epoch 18/20
111600/111600 [==============================] - 444s - loss: 0.1269
Epoch 19/20
111600/111600 [==============================] - 446s - loss: 0.1192
Epoch 20/20
111600/111600 [==============================] - 438s - loss: 0.1124

Car steered crazily to the right side. It looks to me like this was the
straight-ahead right/left copy that thought the model that.

### Ideas to improve

These two are correlated
- Try if it improves without filtering the straight driving
- Try if it improves without left and right images

- Make more data?
- More dropout? Less dropout? Currently we drop 10 % twice
- Is the shuffling really working? Is it important?

### Bad ideas
- Increase batch size a bit to squeeze out more performance. 200 works,
  1000 does not. Naah nvidia-smi shows 3805MiB /  4036MiB



### Best model so far
Epoch 1/10  78400/78400 [==============================] - 259s - loss: 0.2029 - val_loss: 0.1169
Epoch 2/10  78400/78400 [==============================] - 201s - loss: 0.1715 - val_loss: 0.1162
Epoch 3/10  78400/78400 [==============================] - 200s - loss: 0.1792 - val_loss: 0.1155
Epoch 4/10  78400/78400 [==============================] - 200s - loss: 0.1665 - val_loss: 0.1151
Epoch 5/10  78400/78400 [==============================] - 201s - loss: 0.1613 - val_loss: 0.1168
Epoch 6/10  78400/78400 [==============================] - 200s - loss: 0.1472 - val_loss: 0.0888
Epoch 7/10  78400/78400 [==============================] - 200s - loss: 0.1270 - val_loss: 0.0739
Epoch 8/10  78400/78400 [==============================] - 200s - loss: 0.1250 - val_loss: 0.0853
Epoch 9/10  78400/78400 [==============================] - 199s - loss: 0.1194 - val_loss: 0.0658
Epoch 10/10 78400/78400 [==============================] - 199s - loss: 0.1110 - val_loss: 0.0670

With left and right images, flipped and moar data
Drives pretty okay but failed on bridge + dirtcorner (is too far left
before the curve starts)

Without left and right it sucks

### New best (by training longer)
78400/78400             [==============================] - 201s - loss: 0.1809 - val_loss: 0.1175
Epoch 2/20 78400/78400  [==============================] - 196s - loss: 0.1622 - val_loss: 0.1160
Epoch 3/20 78400/78400  [==============================] - 197s - loss: 0.1667 - val_loss: 0.1255
Epoch 4/20 78400/78400  [==============================] - 196s - loss: 0.1656 - val_loss: 0.1154
Epoch 5/20 78400/78400  [==============================] - 196s - loss: 0.1639 - val_loss: 0.1212
Epoch 6/20 78400/78400  [==============================] - 196s - loss: 0.1636 - val_loss: 0.1157
Epoch 7/20 78400/78400  [==============================] - 196s - loss: 0.1443 - val_loss: 0.0874
Epoch 8/20 78400/78400  [==============================] - 196s - loss: 0.1363 - val_loss: 0.0791
Epoch 9/20 78400/78400  [==============================] - 196s - loss: 0.1250 - val_loss: 0.0785
Epoch 10/20 78400/78400 [==============================] - 196s - loss: 0.1224 - val_loss: 0.0748
Epoch 11/20 78400/78400 [==============================] - 196s - loss: 0.1167 - val_loss: 0.0799
Epoch 12/20 78400/78400 [==============================] - 196s - loss: 0.1212 - val_loss: 0.0767
Epoch 13/20 78400/78400 [==============================] - 196s - loss: 0.1168 - val_loss: 0.0679
Epoch 14/20 78400/78400 [==============================] - 196s - loss: 0.1149 - val_loss: 0.0676
78400/78400             [==============================] - 196s - loss: 0.1102 - val_loss: 0.0663 Epoch 16/20
78400/78400             [==============================] - 196s - loss: 0.1106 - val_loss: 0.0638 Epoch 17/20
78400/78400             [==============================] - 196s - loss: 0.1093 - val_loss: 0.0664
Epoch 18/20 78400/78400 [==============================] - 196s - loss: 0.1089 - val_loss: 0.0663
Epoch 19/20 78400/78400 [==============================] - 196s - loss: 0.1078 - val_loss: 0.0637
Epoch 20/20 78400/78400 [==============================] - 196s - loss: 0.1078 - val_loss: 0.0642

### slight improvement with relu instead of tanh and without dropout
Epoch 2/25 78400/78400 [==============================] - 186s - loss: 0.1204 - val_loss: 0.0823
Epoch 3/25 78400/78400 [==============================] - 185s - loss: 0.1162 - val_loss: 0.0640
Epoch 4/25 78400/78400 [==============================] - 185s - loss: 0.1120 - val_loss: 0.0757
Epoch 5/25 78400/78400 [==============================] - 185s - loss: 0.1073 - val_loss: 0.0615
Epoch 6/25 78400/78400 [==============================] - 185s - loss: 0.1089 - val_loss: 0.0714
Epoch 7/25 78400/78400 [==============================] - 185s - loss: 0.1074 - val_loss: 0.0625
Epoch 8/25 78400/78400 [==============================] - 185s - loss: 0.1060 - val_loss: 0.0667
Epoch 9/25 78400/78400 [==============================] - 185s - loss: 0.1021 - val_loss: 0.0610
Epoch 10/25 78400/78400 [==============================] - 184s - loss: 0.0997 - val_loss: 0.0602
Epoch 11/25 78400/78400 [==============================] - 184s - loss: 0.0979 - val_loss: 0.0631
Epoch 12/25 78400/78400 [==============================] - 184s - loss: 0.0974 - val_loss: 0.0613
Epoch 13/25 78400/78400 [==============================] - 184s - loss: 0.0944 - val_loss: 0.0610
Epoch 14/25 78400/78400 [==============================] - 184s - loss: 0.0944 - val_loss: 0.0870
Epoch 15/25 78400/78400 [==============================] - 184s - loss: 0.0940 - val_loss: 0.0601
Epoch 16/25 78400/78400 [==============================] - 184s - loss: 0.0929 - val_loss: 0.0590
Epoch 17/25 78400/78400 [==============================] - 184s - loss: 0.1036 - val_loss: 0.0627
Epoch 18/25 78400/78400 [==============================] - 184s - loss: 0.0949 - val_loss: 0.0616
Epoch 19/25 78400/78400 [==============================] - 186s - loss: 0.0909 - val_loss: 0.0600
Epoch 20/25 78400/78400 [==============================] - 184s - loss: 0.0898 - val_loss: 0.0597
Epoch 21/25 78400/78400 [==============================] - 188s - loss: 0.0890 - val_loss: 0.0689
Epoch 22/25 78400/78400 [==============================] - 189s - loss: 0.0887 - val_loss: 0.0596
Epoch 23/25 78400/78400 [==============================] - 190s - loss: 0.0877 - val_loss: 0.0604
Epoch 24/25 78400/78400 [==============================] - 191s - loss: 0.0862 - val_loss: 0.0592
Epoch 25/25 78400/78400 [==============================] - 190s - loss: 0.0849 - val_loss: 0.0603

got past the dirt corner but went off course afterwards

### full round again with only one touch of the line
78400/78400 [==============================] - 150s - loss: 0.1394 - val_loss: 0.0840
Epoch 2/25
78400/78400 [==============================] - 150s - loss: 0.1204 - val_loss: 0.0813
Epoch 3/25
78400/78400 [==============================] - 150s - loss: 0.1163 - val_loss: 0.0723
Epoch 4/25
78400/78400 [==============================] - 150s - loss: 0.1112 - val_loss: 0.0675
Epoch 5/25
78400/78400 [==============================] - 149s - loss: 0.1105 - val_loss: 0.0734
Epoch 6/25
78400/78400 [==============================] - 150s - loss: 0.1139 - val_loss: 0.0847
Epoch 7/25
78400/78400 [==============================] - 149s - loss: 0.1094 - val_loss: 0.0655
Epoch 8/25
78400/78400 [==============================] - 149s - loss: 0.1062 - val_loss: 0.0687
Epoch 9/25
78400/78400 [==============================] - 149s - loss: 0.1048 - val_loss: 0.0662
Epoch 10/25
78400/78400 [==============================] - 149s - loss: 0.1024 - val_loss: 0.0679
Epoch 11/25
78400/78400 [==============================] - 149s - loss: 0.1009 - val_loss: 0.0659
Epoch 12/25
78400/78400 [==============================] - 149s - loss: 0.0987 - val_loss: 0.0665
Epoch 13/25
78400/78400 [==============================] - 149s - loss: 0.0992 - val_loss: 0.0652
Epoch 14/25
78400/78400 [==============================] - 149s - loss: 0.0967 - val_loss: 0.0650
Epoch 15/25
78400/78400 [==============================] - 149s - loss: 0.0947 - val_loss: 0.0658
Epoch 16/25
78400/78400 [==============================] - 149s - loss: 0.0923 - val_loss: 0.0650
Epoch 17/25
78400/78400 [==============================] - 148s - loss: 0.0935 - val_loss: 0.0631
Epoch 18/25
78400/78400 [==============================] - 149s - loss: 0.0945 - val_loss: 0.0644
Epoch 19/25
78400/78400 [==============================] - 148s - loss: 0.0921 - val_loss: 0.0637
Epoch 20/25
78400/78400 [==============================] - 149s - loss: 0.0900 - val_loss: 0.0634
Epoch 21/25
78400/78400 [==============================] - 148s - loss: 0.0905 - val_loss: 0.0613
Epoch 22/25
78400/78400 [==============================] - 147s - loss: 0.0884 - val_loss: 0.0628
Epoch 23/25
78400/78400 [==============================] - 148s - loss: 0.0932 - val_loss: 0.0601
Epoch 24/25
78400/78400 [==============================] - 148s - loss: 0.0883 - val_loss: 0.0608
Epoch 25/25
78400/78400 [==============================] - 147s - loss: 0.0876 - val_loss: 0.0666
