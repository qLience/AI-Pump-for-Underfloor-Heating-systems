import collections
import random
from collections import deque

Transition = collections.namedtuple("Transition", "state action reward next_state")


class NStepTransition:
    def __init__(self, transitions):
        if not isinstance(transitions, deque):
            self.transitions = [transitions]
            self.n = 1
        else:
            self.transitions = list(transitions)
            self.n = len(transitions)

    def __getitem__(self, item):
        return self.transitions.__getitem__(item)

    def __len__(self):
        return len(self.transitions)

class NStepReplayMemory:
    def __init__(self, capacity, n):
        self.n = n
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        if event.n != self.n:
            raise ValueError("Must be NStepTransition with n : %i" % self.n)
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
