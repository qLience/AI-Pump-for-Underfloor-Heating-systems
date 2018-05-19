import os
import pickle


class SaveOrchestrator:
    def __init__(self, save_dir, brain):
        self.brain = brain
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def save_brain(self, filename):
        self.brain.save(filename)

    def load_brain(self, filename):
        self.brain.load(filename)