import os
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D


class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)
        self.js = 0


    def _build_model(self, num_layers, width):
        """
        全连通深度神经网络的建立与编译
        """
        inputs_1 = keras.Input(shape=(16, 12, 1))
        x1 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(inputs_1)
        x1 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x1)
        x1 = Flatten()(x1)


        inputs_2 = keras.Input(shape=(16, 12, 1))
        x2 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(inputs_2)
        x2 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x2)
        x2 = Flatten()(x2)

        inputs_3 = keras.Input(shape=(16, 12, 1))
        x3 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(inputs_3)
        x3 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x3)
        x3 = Flatten()(x3)

        x = keras.layers.concatenate([x1, x2, x3])
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(self._output_dim, activation='linear')(x)

        # outputs = layers.Dense(self._output_dim, activation='linear')(x)

        model = keras.Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=[x], name='my_model')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))

        return model
    

    def predict_one(self, state):
        """
       从单个状态预测动作值
        """
        return self._model.predict(state)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """

        self._model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)


    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim