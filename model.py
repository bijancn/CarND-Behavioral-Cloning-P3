from __future__ import print_function
import numpy as np
import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import itertools as it
from itertools import *
from keras.models import *
from keras.layers import *


SEED = 1337  # for reproducibility
np.random.seed(SEED)
batch_size = 1000
sample_size = 14668 * 2


def load_image(folder, source_path):
  filename = os.path.basename(source_path)
  new_path = folder + 'IMG/' + filename
  image = cv2.imread(new_path)
  return image # this is in BGR format


def load_csv(folder):
  while True:   # because Keras interface doesn't make sense
    with open(folder + 'driving_log.csv') as csvfile:
      reader = csv.reader(csvfile)
      lines = [l for l in reader]
      lines = lines[1:] # first line was a header in test data
      correction = 0.10
      for l in lines:
        center_angle = float(l[3])
        if abs(center_angle) > 0.001:     # ignore lazy straight driving
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
      yield np.array(acc_x), np.array(acc_y)
      acc_x, acc_y = [], []


def load_data():
  generator_train = load_csv('data/')
  generator_dirt = load_csv('dirtcorner/')
  generator_dirt_extra = load_csv('dirt_extra/')
  generator_extra = load_csv('more/')
  generator = chain(generator_train, generator_dirt,
                    generator_dirt_extra, generator_extra)
  return batch(generator)


def setup_model():
  fraction_to_drop = 0.25
  model = Sequential()
  model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
  model.add(Cropping2D(cropping=((70,25),(0,0))))
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
  model.add(Dense(10))
  model.add(Dense(1))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  return model


def analyze_angles(Y_train):
  sb.distplot(Y_train)
  plt.savefig('dist.png')
  print(np.mean(Y_train))


if __name__ == "__main__":
  generator = load_data()
  #  analyze_angles(Y_train)
  model = setup_model()
  model.fit_generator(generator,
            steps_per_epoch=sample_size/batch_size + 1,
            epochs=30,
            shuffle=True,
            verbose=1)
  model.save('model.h5')
