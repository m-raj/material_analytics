# -*- coding: utf-8 -*-
"""Strain-model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dxverRW1i7LGx-UrNpBsCXi2PWyE0j_i

Downloading dataset
"""

!mv kaggle.json /root/.kaggle/.
!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets download gomayank/rve-elasticity-simulation --unzip

"""The complete RVE is fed as single input to the the neural network."""

import numpy as np
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers, losses
from tensorflow import keras
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.models import load_model
import math

NUM_CLASSES = 6
NUM_ELEMENTS = 31*31*31
VOLUME_FRACS = (10, 30)
NUM_RVE = 100
E33_MEAN_STD = (0.001, 4.4231E-4)
DATASET_PATH = '/content/'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Supress warnings

rve_index = np.arange(NUM_RVE)
train_rves, test_rves = train_test_split(rve_index)

class RVE_Dataset(Sequence):
    def __init__(self,
                 rve_list,
                 batch_size=32,
                 dataset_path=DATASET_PATH,
                 num_classes=NUM_CLASSES,
                 vol_fracs=VOLUME_FRACS,
                 mean_std=E33_MEAN_STD
                 ):
        self.phase_dist = None
        self.target_value = None
        self.current_class = None
        self.current_volume_fraction = None

        self.rve_list = rve_list
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.vol_fracs = vol_fracs
        self.mean_std = mean_std

        self._files_read = list()
        self._num_samples = self.num_classes*len(self.vol_fracs)*len(self.rve_list)

    def __len__(self):
        return math.ceil(self._num_samples/self.batch_size)

    def __getitem__(self, index):
        indeces = np.arange(index * self.batch_size, min(self._num_samples, (index+1) * self.batch_size))
        x = []
        y = []
        # print('\n',indeces)
        for i in indeces:
            if not(i % len(self.rve_list)):
                self._set_current_variables(i // len(self.rve_list))
                # print(i // len(self.rve_list), self.current_volume_fraction, self.current_class)
                x_meta_data = self._set_phase_distribution()
                y_meta_data = self._set_target_variable()
                if x_meta_data == y_meta_data:
                    self._files_read.append(x_meta_data)
            single_x, single_y = self._get_single_item(i % len(self.rve_list))
            x.append(single_x)
            y.append(single_y)
        return np.asarray(x), self._preprocess_target(np.asarray(y))
    
    def _get_single_item(self, index):
        x = np.expand_dims(self.phase_dist[:, self.rve_list[index]].reshape(31, 31, 31), axis=-1)
        y = np.expand_dims(self.target_value[:, self.rve_list[index]].reshape(31, 31, 31), axis = -1)
        return x, y

    def _set_target_variable(self):
        path = self.dataset_path + 'vol_frac_'+str(self.current_volume_fraction) +\
                '/E33/E33_class_' + str(self.current_class)+'.npy'
        self.target_value = np.load(path)
        return 'Vol frac: {0}; Class: {1}'.format(self.current_volume_fraction, self.current_class)

    def _set_current_variables(self, index):
        self.current_class = (index % self.num_classes) + 1
        index = index//self.num_classes
        self.current_volume_fraction = self.vol_fracs[index % len(self.vol_fracs)]

    def _set_phase_distribution(self):
        path = self.dataset_path + 'vol_frac_'+str(self.current_volume_fraction) +\
                '/phase_distribution/class_' + \
                str(self.current_class)+'.npy'
        self.phase_dist = np.load(path)
        return 'Vol frac: {0}; Class: {1}'.format(self.current_volume_fraction, self.current_class)

    def _preprocess_target(self, y):
        return (y-self.mean_std[0])/self.mean_std[1]

    def files_read(self):
        for element in self._files_read:
            print(element)

class Sirius(Model):
    def train_step(self, inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            y_pred = self.call(x)
            loss = self.compiled_loss(y, y_pred)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

fixed_input = layers.Input(shape=(31, 31, 31, 1), name='fixed_input')
x = layers.Conv3D(120, (11, 11, 11),
                  activation=keras.activations.relu,
                  name='conv1')(fixed_input)
x = layers.Conv3D(120, (3, 3, 3),
                  activation=keras.activations.relu,
                  name='conv2')(x)
x = layers.Conv3DTranspose(1, (3, 3, 3),
                                activation = keras.activations.relu)(x)
output = layers.Conv3DTranspose(1, (11, 11, 11),
                                activation = keras.activations.sigmoid)(x)
model = Sirius(inputs=fixed_input, outputs=output)

model.compile(loss=losses.MeanSquaredError(),
              optimizer=optimizers.Adam(learning_rate=1E-3))

callbacks = [keras.callbacks.ReduceLROnPlateau(moniter="val_loss",
                                               factor=0.8,
                                               patience=5,
                                               min_delta=0.001,
                                               cooldown=10,
                                               verbose=1,
                                               min_lr=5E-5)]

tic = time.process_time()
train_subset = RVE_Dataset(train_rves, batch_size=30)
test_subset = RVE_Dataset(test_rves, batch_size=30)
model.fit(train_subset, epochs=10, callbacks=callbacks, workers = 1, use_multiprocessing=False, shuffle=False)
toc = time.process_time()
print('Training completed in {0} seconds'.format(toc-tic))

