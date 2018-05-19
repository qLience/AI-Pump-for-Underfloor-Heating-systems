import collections
import os
import random
import shutil

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

UPDATE_EVERY_N_STEPS = 5000

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


class QNet:
    def __init__(self, input_size, nb_action, prefix):
        self.prefix = prefix
        self.input_tensor = tf.placeholder(shape=[None, input_size], dtype=tf.float32)

        self.fc1 = slim.fully_connected(inputs=self.input_tensor, num_outputs=HIDDEN_LAYER_SIZE,
                                        activation_fn=tf.nn.relu, scope=prefix + "/fc1")
        self.fc2 = slim.fully_connected(inputs=self.fc1, num_outputs=HIDDEN_LAYER_SIZE, activation_fn=tf.nn.relu,
                                        scope=prefix + "/fc2")
        self.q = slim.fully_connected(inputs=self.fc2, num_outputs=nb_action, activation_fn=None, scope=prefix + "/q")
        self.softmax = slim.softmax(self.q * 80, scope=prefix + "/softmax")
        slim.summary.tensor_summary("softmax", self.softmax)
        self.chosen_action = tf.argmax(self.softmax, axis=1)

        self.action = tf.placeholder(shape=[300], dtype=tf.int32)
        self.target = tf.placeholder(shape=[300], dtype=tf.float32)

        self.hot = slim.one_hot_encoding(self.action, nb_action, scope=prefix + "/one_hot")
        self.predictions = tf.reduce_sum(self.hot * self.q, axis=1)
        self.loss = tf.reduce_sum((self.predictions - self.target) ** 2)

        self.optimizer = slim.train.AdamOptimizer()

        self.training = slim.learning.create_train_op(total_loss=self.loss, optimizer=self.optimizer,
                                                      summarize_gradients=True)

        sess = tf.Session()
        self.sess = sess

        self.ctr = 0

        self.summary_op = slim.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('train/', sess.graph)
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init)

    def predict_q_values(self, input_tensor):
        return self.sess.run(self.q, feed_dict={self.input_tensor: input_tensor})

    def select_action(self, new_signal):
        q_orig, softmax, action_value = self.sess.run([self.q, self.softmax, self.chosen_action],
                                                      feed_dict={self.input_tensor: [new_signal]})
        return action_value[0]

    def learn(self, inputs, selected_actions, targets):
        predictions, loss, training, summary = self.sess.run(
            [self.predictions, self.loss, self.training, self.summary_op],
            feed_dict={self.input_tensor: inputs, self.action: selected_actions, self.target: targets})
        self.train_writer.add_summary(summary)

    def save(self, filename):
        self.saver.save(self.sess, filename)

    def load(self, filepath):
        self.saver = tf.train.import_meta_graph(filepath + ".meta")
        # dir, filename = os.path.split(filepath)
        self.saver.restore(self.sess, filepath)

    def update_with_other_network(self, other_network):
        own_vars = tf.trainable_variables(self.prefix)
        other_vars = tf.trainable_variables(other_network.prefix)
        if len(own_vars) != len(other_vars):
            raise AssertionError("Target network must have the same number of trainable vars")
        for i in range(len(own_vars)):
            own_vars[i].assign(other_vars[i])


class Dqn:

    def __init__(self, input_size, nb_action, gamma):
        try:
            shutil.rmtree("train/")
        except OSError:
            print("")
        self.reward_window = []
        self.memory = ReplayMemory(300000)
        self.last_action = 0
        self.last_state = np.zeros(input_size)
        self.num_action = nb_action
        self.gamma = gamma
        self.online = QNet(input_size, nb_action, "online")
        self.target = QNet(input_size, nb_action, "target")
        self.steps_since_last_update = 0

    def calculate_transition_reward(self, transition):
        def calculate_decay_reward(tup):
            r, i = tup
            return self.gamma ** i * r

        return sum(lmap(calculate_decay_reward,
                       zip(lmap(lambda t: t.reward, reversed(transition[:-1])), range(transition.n - 1))))

    def learn_from_transitions(self, transitions):

        states = np.array(lmap(lambda transition: transition[0].state, transitions))

        next_state_qs = self.target.predict_q_values(
            np.array(lmap(lambda transition: transition[-1].next_state, transitions)))

        rewards = np.array(lmap(self.calculate_transition_reward, transitions))
        actions = np.array(lmap(lambda transition: transition[0].action.index, transitions))
        next_max_qs = next_state_qs.max(1)
        target = ((self.gamma ** len(transitions)) * next_max_qs) + rewards
        self.online.learn(states, actions, target)
        if self.steps_since_last_update >= UPDATE_EVERY_N_STEPS:
            print
            "Updating target network"
            self.target.update_with_other_network(self.online)
            self.steps_since_last_update = 0
        else:
            self.steps_since_last_update += 1

    def update(self, new_signal):

        action = self.online.select_action(new_signal)

        return action

    def append_reward(self, reward):
        self.reward_window.append(reward)

    def score(self):
        return sum(self.reward_window) / len(self.reward_window) + 1.

    def save(self, filename):
        self.online.save(filename)

    def load(self, filepath):
        if os.path.exists(filepath):
            print("===>> Loading checkpoint ...")
            filepath, _ = os.path.splitext(filepath)
            self.online.load(filepath)
            self.target.load(filepath)
            print("Loaded checkpoint")
        else:
            print("Nothing to load")
