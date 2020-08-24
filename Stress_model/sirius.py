# %% [code]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers, losses
from tensorflow import keras
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.models import load_model
import math

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
dataset_file_loc = '../../Abaqus_dataset/'
phase_dist = np.load(dataset_file_loc+'phase_dist.npy')
phase_dist = pd.DataFrame(phase_dist, columns=['Element-Id', 'Phase-Id'],dtype='int64').set_index('Element-Id')
frame_description = np.load(dataset_file_loc+'frame_description.npy', encoding='latin1', allow_pickle=True)
element_index = np.load(dataset_file_loc+'element_index.npy', encoding='latin1', allow_pickle=True)
element_arrangement = np.load(dataset_file_loc+'element_arrangement.npy', allow_pickle=True, encoding='latin1')
mean = np.load('../../stress_means.npy')[2]
variance = np.load('../../stress_variance.npy')[2]

U_MAX = 0.05
BATCH_SIZE = 200
EPOCHS = 200
NUM_WORKERS = 10
NEIGHBORS = 4
LOCAL_LEN = 2*NEIGHBORS + 1

# Train test split of the element ids
train_set, test_set = train_test_split(phase_dist.index.values, test_size=0.1)

tf.keras.backend.set_floatx('float64')
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=12,
                                  device_count={'CPU': 20},
                                  intra_op_parallelism_threads=8)
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


class Dataset(Sequence):
    def __init__(self,
                 frame_description,
                 phase_dist,
                 element_arrangement,
                 element_index,
                 subset,
                 mean=mean,
                 var=variance,
                 neighbors=NEIGHBORS,
                 batch_size=32):
        self.frame_description = frame_description
        self.phase_dist = phase_dist
        self.element_arrangement = element_arrangement
        self.element_index = element_index
        self.batch_size = batch_size
        self.mean = mean
        self.std = var**0.5
        self.neighbors = neighbors
        self.subset = subset

    def __len__(self):
        return math.ceil(len(self.subset) / self.batch_size)

    def __getitem__(self, index):
        element_ids = self.subset[index *
                                  self.batch_size:(index + 1)
                                  * self.batch_size]
        target = []
        local_phase_dist = []
        for element_id in element_ids:
            path = "../../Stress/"+str(element_id-1)+'.npy'
            target.append(np.load(path)[200, 3])  # Sigma33
            # geometrical features
            local_phase_dist.append(self._get_neighbors(element_id))
        local_phase_dist = np.asarray(local_phase_dist)
        target = np.asarray(target)
        return self._preprocess_local_phase_dist(local_phase_dist), self._preprocess_target(target)

    def _preprocess_local_phase_dist(self, x):
        return x-0.5

    def _preprocess_target(self, target):
        return (target-self.mean)/self.std

    def _get_neighbors(self, element_id):
        element_coords = self.element_index[int(element_id-1)]
        lower_limits = np.zeros(3)
        for i in range(3):
            margin = element_coords[i]-self.neighbors
            if not(margin < 0):
                lower_limits[i] = int(margin)
            else:
                lower_limits[i] = 0
        upper_limits = np.zeros(3)
        for i in range(3):
            margin = element_coords[i]+self.neighbors+1
            if margin > 30:
                upper_limits[i] = 30
            else:
                upper_limits[i] = int(margin)
        x_min, y_min, z_min = lower_limits
        x_max, y_max, z_max = upper_limits
        local_arangement = self.element_arrangement[int(x_min):int(x_max), int(
            y_min):int(y_max), int(z_min): int(z_max)].copy()
        I, J, K = local_arangement.shape
        L = 2*self.neighbors+1
        local_phase_distribution = np.ones((L, L, L))*0.3  #0.3 due to 30% class
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    local_phase_distribution[i, j, k] = self.phase_dist.loc[int(local_arangement[i, j, k]), 'Phase-Id']
        return local_phase_distribution


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


fixed_input = layers.Input(shape=(LOCAL_LEN, LOCAL_LEN, LOCAL_LEN), name='fixed_input')
x = layers.Conv2D(12, (3, 3),
                  activation=keras.activations.relu,
                  name='conv1')(fixed_input)
x = layers.Conv2D(12, (3, 3),
                  activation=keras.activations.relu,
                  name='conv2')(x)
x = layers.MaxPooling2D((2,2), name='Pooling')(x)
x = layers.Flatten(name='flatten')(x)
x = layers.Dropout(rate=0.2, name='dense_dropout')(x)
x = layers.Dense(50, name='dense1', activation=keras.activations.tanh)(x)
output = layers.Dense(1, name='output')(x)
model = Sirius(inputs=fixed_input, outputs=output)

train_dataset = Dataset(frame_description,
                        phase_dist, element_arrangement,
                        element_index,
                        train_set,
                        batch_size=BATCH_SIZE)

test_dataset = Dataset(frame_description,
                       phase_dist, element_arrangement,
                       element_index,
                       test_set,
                       batch_size=BATCH_SIZE)

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
logs = model.fit(train_dataset,
                 validation_data=test_dataset,
                 callbacks=callbacks,
                 epochs=EPOCHS,
                 workers=NUM_WORKERS,
                 use_multiprocessing=True)
toc = time.process_time()
print('Process time (seconds):\t', toc-tic)

# Save logs
for key in logs.history.keys():
    np.save(key, logs.history[key])

# Save model
model.save('model')
