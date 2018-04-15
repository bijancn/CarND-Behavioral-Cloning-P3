from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from keras.models import *
from keras.layers import *


def load_image(folder, source_path):
  filename = os.path.basename(source_path)
  new_path = folder + 'IMG/' + filename
  image = cv2.imread(new_path)
  return image # this is in BGR format


def load_csv(folder):
  lines = []
  with open(folder + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    lines = [l for l in reader]
    lines = lines[1:] # first line was a header in test data
    correction = 0.10
    images = []
    angles = []
    for l in lines:
      center_angle = float(l[3])
      if abs(center_angle) > 0.001:     # ignore lazy straight driving
        left_angle = center_angle + correction
        right_angle = center_angle - correction
        img_center = load_image(folder, l[0])
        img_left = load_image(folder, l[1])
        img_right = load_image(folder, l[2])
        images.append(img_center)
        images.append(img_left)
        images.append(img_right)
        angles.append(center_angle)
        angles.append(left_angle)
        angles.append(right_angle)
  return np.array(images), np.array(angles)

def setup_model():
  model = Sequential()
  model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
  model.add(Cropping2D(cropping=((70,25),(0,0))))
  # 5 conv nets
  model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
  model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
  model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
  model.add(Convolution2D(64,3,3,activation="relu"))
  model.add(Convolution2D(64,3,3,activation="relu"))
  model.add(Flatten())
  # 4 fully connected
  model.add(Dense(100))
  model.add(Dense(50))
  model.add(Dense(10))
  model.add(Dense(1))
  model.compile(loss='mean_absolute_error', optimizer='adam')
  return model


def analyze_angles(Y_train):
  sb.distplot(Y_train)
  plt.savefig('dist.png')
  print(np.mean(Y_train))


if __name__ == "__main__":
  X_train, Y_train = load_csv('data/')
  X_dirt, Y_dirt = load_csv('dirtcorner/')
  X_dirt_extra, Y_dirt_extra = load_csv('dirt_extra/')
  X_extra, Y_extra = load_csv('more/')
  X_train = np.append(X_train, X_dirt, axis=0)
  Y_train = np.append(Y_train, Y_dirt, axis=0)
  X_train = np.append(X_train, X_extra, axis=0)
  Y_train = np.append(Y_train, Y_extra, axis=0)
  X_train = np.append(X_train, X_dirt_extra, axis=0)
  Y_train = np.append(Y_train, Y_dirt_extra, axis=0)

  analyze_angles(Y_train)

  model = setup_model()
  model.fit(X_train, Y_train,
            validation_split=0.2,
            shuffle=True,
            epochs=3,
            verbose=1)
  model.save('model.h5')
