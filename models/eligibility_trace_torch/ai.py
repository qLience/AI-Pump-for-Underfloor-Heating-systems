# This code have been programmed by Christian Blad, Sajuran and Søren Koch aka. Group VT4103A
# The reinforcement learning framework is based on the original code from udemy course https://www.udemy.com/artificial-intelligence-az/
# From Aalborg University

# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Making the body
class SoftmaxBody(nn.Module):
    
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T

    def forward(self, outputs):
        """Returns action randomly drawn from a distribution of Q-values where
        state is send through Q-network"""
        probs = F.softmax(outputs * self.T)   
        actions = probs.multinomial()
        return actions

# Making the AI
class AI:

    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()


class Training:
    def __init__(self, params, ai, eligibility_memory, n_steps, dqn):
        self.params = params
        self.nb_epochs = 0
        self.optimizer = optim.Adam(ai.brain.parameters(), params.lr)
        self.loss = nn.MSELoss()
        self.eligibility_memory = eligibility_memory
        self.ai = ai
        self.n_steps = n_steps
        self.reward_window = []
        self.scores = []
        
    def update(self):
        """This function update/links the learning, experience list, action selector, Q-network and the reward window"""
        # Run steps
        self.eligibility_memory.run_steps(1)
        
        # Learn when it have enough in buffer and every 20 times inorder not to have a to high execution time of iteration
        if  len(self.eligibility_memory.buffer) > self.params.ER_batch_size and  len(self.eligibility_memory.buffer) % 100 == 0 and self.params.learning_mode == 1:
            self.learn()
        
        # Make window for brain plot
        self.reward_window.append(self.n_steps.rewardski)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
            
        #update score
        self.scores.append(self.score())

    def score(self):
        """Returns sum of rewards"""
        return sum(self.reward_window)/(len(self.reward_window)+1.)

	# Saving experience
    def save_plot(self, path, name):
        """Saving training curves tracking the agent score to the path 'saves/plot' with the specified name specified in parse argument in main"""
        plt.plot(self.scores, color='red')
        plt.ylabel('reward score per iterations')
        plt.xlabel('Training iterations')
        plt.title('Training curves tracking the agent score')
        plt.savefig(os.path.join(path, str(name) + '.pdf'), format='pdf')
    
        
    def eligibility_trace(self, batch):
        """Perform eligibility trace"""
        inputs = []
        targets = []
        for series in batch:
            input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
            output = self.ai.brain(input)
            cumul_reward = output[1].data.max()
            for step in reversed(series[:-1]):
                cumul_reward = step.reward + self.params.gamma * cumul_reward
            state = series[0].state
            target = output[0].data
            target[series[0].action] = cumul_reward
            inputs.append(state)
            targets.append(target)
        return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)
    
            
    def learn(self): 
        """Updating weights of Q-network depending on the loss calculated from the predicted Q-values
        from the Q-network and the target Q-values(gamma*Qp + r)"""
        for batch in self.eligibility_memory.sample_batch(self.params.ER_batch_size):
            inputs, targets = self.eligibility_trace(batch)
            inputs, targets = Variable(inputs), Variable(targets)
            predictions = self.ai.brain(inputs)
            loss_error = self.loss(predictions, targets)
            self.optimizer.zero_grad()
            loss_error.backward()
            self.optimizer.step()
        rewards_steps = self.n_steps.rewards_steps()
        
        
    def save_brain(self, path, name):
        torch.save({'state_dict': self.ai.brain.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, os.path.join(path, str(name) + '.pth'))
    
    def load_brain(self, path, name):
        if os.path.isfile(os.path.join(path, str(name) + '.pth')):
            print("=> loading brain... ")
            checkpoint = torch.load(os.path.join(path, str(name) + '.pth'))
            self.ai.brain.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loading is complete !")
        else:
            print("no brain found...")
            exit(1)