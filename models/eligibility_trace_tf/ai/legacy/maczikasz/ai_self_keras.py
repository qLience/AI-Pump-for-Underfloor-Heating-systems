import collections
import os
import random

import keras
import numpy as np
from future.utils import lmap
from keras import Sequential
from keras import backend as K
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras.models import load_model
from keras.optimizers import Adam

HIDDEN_LAYER_SIZE = 30

Transition = collections.namedtuple("Transition", "state action reward next_state")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        raise AssertionError("Keras implementation longer supported doe to eligibility trace")

        self.gamma = gamma
        self.reward_window = []
        self.gamma = gamma
        self.memory = ReplayMemory(300000)
        self.last_action = 0
        self.last_state = np.zeros(input_size)
        self.num_action = nb_action
        self.model = Sequential()
        self.model.add(Dense(units=30, activation='relu', input_dim=input_size))
        self.model.add(Dense(units=30, activation='relu'))
        self.model.add(Dense(units=nb_action))

        adam = Adam()
        self.model.compile(adam, loss=mean_squared_error)

    def _learn(self, transitions):
        states = np.array(lmap(lambda transition: transition.state, transitions))

        next_stateQs = self.model.predict(np.array(lmap(lambda transition: transition.next_state, transitions)))

        rewards = np.array(lmap(lambda transition: transition.reward, transitions))
        actions = np.array(lmap(lambda transition: transition.action, transitions))
        next_max_qs = next_stateQs.max(1)
        target = (self.gamma * next_max_qs) + rewards
        one_hot = keras.utils.to_categorical(actions, self.num_action)

        self.model.fit(states, (one_hot.T * target).T, verbose=0)

    def update(self, reward, new_signal):
        self.memory.push(
            Transition(state=self.last_state, action=self.last_action, reward=reward, next_state=new_signal))

        qvalues = self.model.predict(np.expand_dims(np.array(new_signal), 1).T)
        action = K.eval(K.argmax(K.softmax(qvalues), 1))[0]

        if len(self.memory.memory) > 300:
            transitions = self.memory.sample(100)
            self._learn(transitions)

        self.reward_window.append(reward)
        self.last_state = new_signal
        self.last_action = action

        return action

    def score(self):
        return sum(self.reward_window) / len(self.reward_window) + 1.

    def save(self, filename):
        self.model.save()

    def load(self, filename):
        if os.path.exists(filename):
            print("===>> Loading checkpoint ...")
            self.model = load_model(filename)
        else:
            print("Nothing to load")
