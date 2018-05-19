import os
import shutil

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from future.utils import lmap

HIDDEN_LAYER_SIZE = 30


class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        try:
            shutil.rmtree("train/")
        except OSError:
            print("")
        self.reward_window = []
        self.gamma = gamma
        self.last_action = 0
        self.last_state = np.zeros(input_size)
        self.num_action = nb_action

        self.input_tensor = tf.placeholder(shape=[None, input_size], dtype=tf.float32)

        self.fc1 = slim.fully_connected(inputs=self.input_tensor, num_outputs=30, activation_fn=tf.nn.relu, scope="fc1")
        self.fc2 = slim.fully_connected(inputs=self.fc1, num_outputs=30, activation_fn=tf.nn.relu, scope="fc2")
        self.q = slim.fully_connected(inputs=self.fc2, num_outputs=nb_action, activation_fn=None, scope="q")
        self.softmax = slim.softmax(self.q * 70, scope="softmax")
        slim.summary.tensor_summary("softmax", self.softmax)
        self.chosen_action = tf.argmax(self.softmax, axis=1)

        self.action = tf.placeholder(shape=[300], dtype=tf.int32)
        self.target = tf.placeholder(shape=[300], dtype=tf.float32)

        self.hot = slim.one_hot_encoding(self.action, self.num_action, scope="one_hot")
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

    def calculate_transition_reward(self, transition):
        def decay_reward(tup):
            r, i = tup
            return self.gamma ** i * r

        return sum(lmap(decay_reward, zip(lmap(lambda t: t.reward, reversed(transition[:-1])), range(transition.n - 1))))

    def learn_from_transitions(self, transitions):
        states = np.array(lmap(lambda transition: transition[0].state, transitions))

        next_stateQs = self.sess.run(self.q, feed_dict={
            self.input_tensor: np.array(lmap(lambda transition: transition[-1].next_state, transitions))})

        rewards = np.array(lmap(self.calculate_transition_reward, transitions))
        actions = np.array(lmap(lambda transition: transition[0].action.index, transitions))
        next_max_qs = next_stateQs.max(1)
        target = ((self.gamma ** len(transitions)) * next_max_qs) + rewards

        predictions, loss, training, summary = self.sess.run(
            [self.predictions, self.loss, self.training, self.summary_op],
            feed_dict={
                self.input_tensor: states,
                self.action: actions,
                self.target: target})
        self.train_writer.add_summary(summary)

    def update(self, new_signal):
        q_orig, softmax, action_value = self.sess.run([self.q, self.softmax, self.chosen_action],
                                                      feed_dict={self.input_tensor: [new_signal]})
        action = action_value[0]

        return action

    def append_reward(self, reward):
        self.reward_window.append(reward)

    def score(self):
        return sum(self.reward_window) / len(self.reward_window) + 1.

    def save(self, filename):
        self.saver.save(self.sess, filename)

    def load(self, filepath):
        if os.path.exists(filepath):
            print("===>> Loading checkpoint ...")
            filepath, _ = os.path.splitext(filepath)
            self.saver = tf.train.import_meta_graph(filepath + ".meta")
            # dir, filename = os.path.split(filepath)
            self.saver.restore(self.sess, filepath)
            print("Loaded checkpoint")
        else:
            print("Nothing to load")
