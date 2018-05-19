from abc import ABCMeta, abstractmethod

class AI:
    def __init__(self, params, dqn_initializator):
        self.brain = dqn_initializator(params)

    def get_next_action(self, input):
        return self.brain.update(input)

    def score(self):
        return self.brain.score()


class AiAction:
    __metaclass__ = ABCMeta
