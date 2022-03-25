import numpy as np
import keras
import random
from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import losses
from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D


class DoubleDQN():
    def __init__(self):

        # hyperparameters
        # self.env = env
        self.gamma = 0.99
        self.nn_learning_rate = 0.0002
        self.nn_batch_size = None
        self.epochs = 1
        self.minibatch_sz = 64
        self.epsilon = 1.
        self.epsilon_decay = 0.992
        self.epsilon_floor = 0.05
        # self.n_s = env.observation_space.shape[0]
        self.n_a = 4  # (输出动作空间)
        self._output_dim = 4

        self.description = 'DQN Learner'
        self.update_frequency = 100
        self.verbose = False

        # memory
        self.memory_max = 50000
        self.reset()

    def reset(self):
        self.epsilon = 1.
        self.step = 0
        self.memory = [[], [], [], []]

        # create nn's
        self.model = self._make_model_()
        self.target_model = self._make_model_()

    def _make_model_(self):

        inputs_1 = keras.Input(shape=(16, 16, 1))
        x1 = Conv2D(16, (7, 7), strides=2, activation='relu')(inputs_1)
        x1 = Conv2D(32, (2, 2), strides=1, activation='relu')(x1)
        x1 = Flatten()(x1)

        inputs_2 = keras.Input(shape=(16, 16, 1))
        x2 = Conv2D(16, (7, 7), strides=2, activation='relu')(inputs_2)
        x2 = Conv2D(32, (2, 2), strides=1, activation='relu')(x2)
        x2 = Flatten()(x2)

        inputs_3 = keras.Input(shape=(16, 16, 1))
        x3 = Conv2D(16, (7, 7), strides=2, activation='relu')(inputs_3)
        x3 = Conv2D(32, (2, 2), strides=1, activation='relu')(x3)
        x3 = Flatten()(x3)

        x = keras.layers.concatenate([x1, x2, x3])
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(self._output_dim, activation='linear')(x)

        # outputs = layers.Dense(self._output_dim, activation='linear')(x)

        model = keras.Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=[x], name='my_model')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=0.001))

        return model

    def _model_update_(self):
        self.target_model.set_weights(self.model.get_weights())

    def pick_action(self, state):

        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_a)
        else:
            tmp = self.model.predict(state)
            return np.argmax(tmp[0])

    def update(self, old_state, old_action, reward, current_sate, done, epsilon):
        memory_len = len(self.memory[0]) + len(self.memory[0]) + len(self.memory[0]) + len(self.memory[0])

        # if len(self.memory) > self.memory_max: self.memory.pop(0)
        if memory_len > self.memory_max: self.memory = [[], [], [], []]

        self.memory[old_action].append([old_state, old_action, reward, current_sate, done])

        self._batch_train_()

        if self.step % self.update_frequency == 0:
            self._model_update_()

        if done and self.epsilon > self.epsilon_floor:
            # self.epsilon = self.epsilon * self.epsilon_decay
            self.epsilon = epsilon

        self.step += 1
    def _batch_train_(self):
        # memory_len = len(self.memory[0]) + len(self.memory[1]) + len(self.memory[2]) + len(self.memory[3])
        m0 = len(self.memory[0])
        m1 = len(self.memory[1])
        m2 = len(self.memory[2])
        m3 = len(self.memory[3])
        # if memory_len > self.minibatch_sz:
        if m1 > 16 and m2 > 16 and m3 > 16 and m0 > 16:
            em1 = np.zeros((100, 16, 16, 1))
            v1 = np.zeros((100, 16, 16, 1))
            p1 = np.zeros((100, 16, 16, 1))
            em2 = np.zeros((100, 16, 16, 1))
            v2 = np.zeros((100, 16, 16, 1))
            p2 = np.zeros((100, 16, 16, 1))
            # create training batch
            # batch = random.sample(self.memory, self.minibatch_sz)
            # print("0", len(self.memory[0]), "1", len(self.memory[1]), "2", len(self.memory[2]), "3", len(self.memory[3]))
            b0 = random.sample(self.memory[0], 16)
            b1 = random.sample(self.memory[1], 16)
            b2 = random.sample(self.memory[2], 16)
            b3 = random.sample(self.memory[3], 16)
            batch = b0 + b1 + b2 + b3

            i = 0
            for val in batch:
                p1[i] = val[0][0]
                em1[i] = val[0][1]
                v1[i] = val[0][2]
                p2[i] = val[3][0]
                em2[i] = val[3][1]
                v2[i] = val[3][2]
                i += 1
            # 根据memory中的随机选择的结果重构丢入网络进行训练
            states = [p1, em1, v1]
            next_states = [p2, em2, v2]

            # use update rule from Minh 2013
            q_s_a = self.model.predict(states)  # predict Q(state), for every sample
            q_s_a_d = self.model.predict(next_states)  # predict Q(next_state), for every sample
            tm_q_s_a_d = self.target_model.predict(next_states)

            a_maxes = np.argmax(q_s_a_d, axis=1)  # 返回q_s_a_d中最大值的索引
            
            y = np.zeros((100, 1, 4))
            em = np.zeros((100, 16, 16, 1))
            v = np.zeros((100, 16, 16, 1))
            p = np.zeros((100, 16, 16, 1))
            for i, b in enumerate(batch):
                state, action, reward, n_states, done = b[0], b[1], b[2], b[3], b[4]
                target = reward
                if not done:
                    Q_target_max = tm_q_s_a_d[i][a_maxes[i]]
                    target += self.gamma * Q_target_max
                current_q = q_s_a[i]
                current_q[action] = target
                p[i] = state[0]
                em[i] = state[1]
                v[i] = state[2]
                y[i] = current_q
            x = [p, em, v]
            self.model.fit(x, y, epochs=1, verbose=False)  # train the NNt