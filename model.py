from __future__ import print_function
import numpy as np
import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sb
import itertools as it
from itertools import *
from keras.models import *
from keras.layers import *
from keras import *
# import theano
# theano.config.openmp = True

SEED = 1337  # for reproducibility
np.random.seed(SEED)
batch_size = 200

def load_image(folder, source_path):
  filename = os.path.basename(source_path)
  new_path = folder + 'IMG/' + filename
  image = cv2.imread(new_path) # this is in BGR format
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image      # this is now RGB just like the fucking drive.py does it


def load_csv(folder):
  with open(folder + 'driving_log.csv') as csvfile:
    print("Loading: ", folder)
    reader = csv.reader(csvfile)
    lines = [l for l in reader]
    lines = lines[1:] # first line was a header in test data
    correction = 0.10
    for l in lines:
      center_angle = float(l[3])
      if abs(center_angle) > 0.0001:     # ignore lazy straight driving
        left_angle = center_angle + correction
        right_angle = center_angle - correction
        img_center = load_image(folder, l[0])
        img_left = load_image(folder, l[1])
        img_right = load_image(folder, l[2])
        yield (img_center, center_angle)
        yield (img_left, left_angle)
        yield (img_right, right_angle)
        yield (np.fliplr(img_center), - center_angle)
        yield (np.fliplr(img_left), - left_angle)
        yield (np.fliplr(img_right), - right_angle)


def batch(generator, size=batch_size):
  acc_x = []
  acc_y = []
  for x, y in generator:
    acc_x.append(x)
    acc_y.append(y)
    if len(acc_x) == size:
      acc_x, acc_y = shuffle(acc_x, acc_y)
      yield np.array(acc_x), np.array(acc_y)
      acc_x, acc_y = [], []

def shuffle(a,b):
  c = list(zip(a, b))
  random.shuffle(c)
  a, b = zip(*c)
  return (a, b)

def batch_endless(generator, size=batch_size):
  gen, gen_backed = tee(generator)
  while True:
    acc_x = []
    acc_y = []
    for x, y in gen:
      acc_x.append(x)
      acc_y.append(y)
      if len(acc_x) == size:
        acc_x, acc_y = shuffle(acc_x, acc_y)
        yield np.array(acc_x), np.array(acc_y)
        acc_x, acc_y = [], []
    gen, gen_backed = tee(gen_backed)


def load_data(batch_func=batch):
  generator_train = load_csv('data/')
  generator_dirt = load_csv('dirtcorner/')
  generator_dirt_extra = load_csv('dirt_extra/')
  generator_extra = load_csv('more/')
  # generator_jungle = load_csv('jungle/')
  generator_moar = load_csv('moar/')
  generator = chain(generator_train,
                    generator_dirt,
                    generator_dirt_extra,
                    generator_extra,
                    #  generator_jungle,
                    generator_moar)
  return batch_func(generator)


def setup_model():
  fraction_to_drop = 0.10
  model = Sequential()
  model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
  # 50 from top, 20 from bottom, nothing from the sides
  model.add(Cropping2D(cropping=((70,25), (0,0))))
  model.add(Dropout(fraction_to_drop, noise_shape=None, seed=SEED))
  # 5 conv nets
  model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
  model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
  model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
  model.add(Convolution2D(64,3,3,activation="relu"))
  model.add(Convolution2D(64,3,3,activation="relu"))
  model.add(Flatten())
  model.add(Dropout(fraction_to_drop, noise_shape=None, seed=SEED))
  # 4 fully connected
  model.add(Dense(100,activation="relu"))
  model.add(Dense(50,activation="relu"))
  model.add(Dense(10,activation="relu"))
  model.add(Dense(1))
  adam = optimizers.Adam(lr=0.0001)
  model.compile(loss='mean_absolute_error', optimizer=adam)
  return model


def analyze_angles(generator, filename):
  length = 0
  Y_train = []
  for x_array, y_array in generator:
    length += len(x_array)
    print(length)
    Y_train += [y_array]
  Y_train = np.vstack(Y_train)
  print ("shape", Y_train.shape)
  sample_size = length
  print ("Mean of steering angles: ", np.mean(Y_train))
  print ("Sample size: ", sample_size)
  sb.distplot(Y_train.flatten())
  plt.savefig(filename)
  return sample_size


if __name__ == "__main__":
  model = setup_model()
  generator = load_data()
  sample_size = analyze_angles(generator, 'dist.png')
  validation_size = analyze_angles(batch(load_csv('extra_round/')), 'validation_dist.png')
  generator = load_data(batch_func=batch_endless)
  validation_generator = batch_endless(load_csv('extra_round/'))
  checkpoint = callbacks.ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=False,
                            mode='auto',
                            period=1)
  earlystop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                      patience=3, verbose=1, mode='auto')
  hist = model.fit_generator(generator,
                      verbose=1,
                      samples_per_epoch=sample_size,
                      nb_epoch=15,
                      validation_data=validation_generator,
                      nb_val_samples=validation_size,
                             callbacks=[checkpoint, earlystop])
  print(hist.history)
  model.save('model.h5')
