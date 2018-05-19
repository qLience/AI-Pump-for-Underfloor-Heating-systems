import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

HIDDEN_LAYER_SIZE = 30


class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(self.input_size, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.fc3 = nn.Linear(HIDDEN_LAYER_SIZE, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(x))
        q_values = self.fc3(x2)
        return q_values


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        raise AssertionError("Pytorch implementation is longer supported doe to eligibility trace")
        self.gamma = gamma
        self.model = Network(input_size, nb_action)
        self.reward_window = []
        self.memory = ReplayMemory(300000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        vals = self.model(Variable(state, volatile=True))
        vals_with_temp = vals * 70
        probs = F.softmax(vals_with_temp)  # T = 7
        action = probs.multinomial()
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        unsqueeze = batch_action.unsqueeze(1)
        self_model = self.model(batch_state)
        gather = self_model.gather(1, unsqueeze)
        outputs = gather.squeeze(1)
        model = self.model(batch_next_state)
        detach = model.detach()
        detach_max = detach.max(1)
        next_outputs = detach_max[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push(
            (self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 300:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window) / len(self.reward_window) + 1.

    def save(self, filename):
        torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, filename)

    def load(self, filename):
        if os.path.exists(filename):
            print("===>> Loading checkpoint ...")
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("Nothing to load")
