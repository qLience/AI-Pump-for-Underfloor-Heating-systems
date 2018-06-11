# This code have been programmed by Christian Blad, Sajuran and Søren Koch aka. Group VT4103A
# The reinforcement learning framework is based on the original code from udemy course https://www.udemy.com/artificial-intelligence-az/
# From Aalborg University

# Importing the libraries
import numpy as np
import random # random samples from different batches (experience replay)
import os # For loading and saving brain
import torch
import math
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # for using stochastic gradient descent
import torch.autograd as autograd # Conversion from tensor (advanced arrays) to avoid all that contains a gradient
# We want to put the tensor into a varaible taht will also contain a
# gradient and to this we need:
from torch.autograd import Variable
# to convert this tensor into a variable containing the tensor and the gradient

# Creating the architecture of the Neural Network
class Network(nn.Module): #inherinting from nn.Module
    
    #Self - refers to the object that will be created from this class
    #     - self here to specify that we're referring to the object
    def __init__(self, params): #[self,input neuroner, output neuroner]
        super(Network, self).__init__() #inorder to use modules in torch.nn
        self.params = params
        # Full connection between different layers of NN
        # Create the same amount hidden layers accordingly to what is specified in parse argument
        if self.params.hidden_layers == 2: # number of hidden layers
            self.fc1 = nn.Linear(params.input_size, params.hidden_size)
            self.fc2 = nn.Linear(params.hidden_size, params.hidden_size)
            self.fc3 = nn.Linear(params.hidden_size, params.action_size) # 30 neurons in hidden layer
        else:
            self.fc1 = nn.Linear(params.input_size, params.hidden_size)
            self.fc2 = nn.Linear(params.hidden_size, params.action_size) # 30 neurons in hidden layer
    
    # For function that will activate neurons and perform forward propagation
    def forward(self, state):
        """Returns Q-values from Q-network by forwarding states through Q-network"""
        # rectifier function        
        if self.params.hidden_layers == 2: # number of hidden layers
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            q_values = self.fc3(x)
        else:
            x = F.relu(self.fc1(state))
            q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay
# We know that RL is based on MDP
# So going from one state(s_t) to the next state(s_t+1)
# We gonna put 100 transition between state into what we call the memory
# So we can use the distribution of experience to make a decision
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity #100 transitions
        self.memory = [] #memory to save transitions
    
    # pushing transitions into memory with append
    #event=transition
    def push(self, event):
        """Function is pushing experience into a list and make sure that list does
        not exceed experience replay capacity"""
        self.memory.append(event)
        if len(self.memory) > self.capacity: #memory only contain 100 events
            del self.memory[0] #delete first transition from memory if there is more that 100
    
    # taking random sample
    def sample(self, batch_size):
        """Reshapes current experience variables so it is ready to be inserted to experience list"""
        #Creating variable that will contain the samples of memory
        #zip =reshape function if list = ((1,2,3),(4,5,6)) zip(*list)= (1,4),(2,5),(3,6)
        #                (state,action,reward),(state,action,reward)  
        samples = zip(*random.sample(self.memory, batch_size))
        #This is to be able to differentiate with respect to a tensor
        #and this will then contain the tensor and gradient
        #so for state,action and reward we will store the seperately into some
        #bytes which each one will get a gradient
        #so that eventually we'll be able to differentiate each one of them
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning

class DQN():
    
    def __init__(self, params):
        self.params = params
        # Sliding window of the evolving mean of the last 100 events/transitions
        self.reward_window = []
        #Creating network with network class
        self.model = Network(params)
        #creating memory with memory class
        #We gonna take 100000 samples into memory and then we will sample from this memory to 
        #to get a snakk number of random transitions
        self.memory = ReplayMemory(params.ER_capacity)
        #creating optimizer (stochastic gradient descent)
        self.optimizer = optim.Adam(self.model.parameters(), lr = params.lr) #learning rate
        #input vector which is batch of input observations
        #by unsqeeze we create a fake dimension to this is
        #what the network expect for its inputs
        #have to be the first dimension of the last_state
        self.last_state = torch.Tensor(params.input_size).unsqueeze(0)
        #Inilizing
        self.last_action = 0
        self.last_reward = 0
        self.steps_done = 0
    
    def softmax_body(self, state):
        """Returns action randomly drawn from a distribution of Q-values where
        state is send through Q-network"""
        q_values= self.model(Variable(state))

        probs = F.softmax((q_values)*self.params.tau,dim=1)
        #create a random draw from the probability distribution created from softmax
        action = probs.multinomial()
        self.steps_done += 1
        return action.data[0,0]
		
    def epsilon_greedy(self, state):
        """Returns action randomly drawn from a distribution of Q-values where
        state is send through Q-network. The highest Q-value will be drawn more
        more frequently as the epsilon greedy policy decays"""
        q_values= self.model(Variable(state))
			
        sample = random.random()
        eps_threshold = self.params.eps_end + (self.params.eps_start - self.params.eps_end) * \
            math.exp(-1. * self.steps_done / self.params.eps_decay)
        if sample > eps_threshold:
            action =  action = q_values.type(torch.FloatTensor).data.max(1)[1].view(1, 1)
            self.steps_done += 1
            return action[0,0]
        else:
            action = Variable(torch.LongTensor([[random.randrange(self.params.action_size)]]))
            self.steps_done += 1
        
        return action.data[0,0]
    
    # See section 5.3 in AI handbook
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        """Updating weights of Q-network depending on the loss calculated from the predicted Q-values
        from the Q-network and the target Q-values(gamma*Qp + r)"""
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        #next input for target see page 7 in attached AI handbook
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.params.gamma*next_outputs + batch_reward
        #Using hubble loss inorder to obtain loss
        td_loss = F.smooth_l1_loss(outputs, target)
        #using  lass loss/error to perform stochastic gradient descent and update weights 
        self.optimizer.zero_grad() #reintialize the optimizer at each iteration of the loop
        #This line of code that backward propagates the error into the NN
        #td_loss.backward(retain_variables = True) #userwarning
        td_loss.backward(retain_graph = True)
		#And this line of code uses the optimizer to update the weights
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        """This function update/links the learning, experience list, action selector, Q-network and the reward window"""
        #Updated one transition and we have dated the last element of the transition
        #which is the new state
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
		
		#After ending in a state its time to play a action
        if self.params.action_selector == 1: #Softmax
            action = self.softmax_body(new_state)
        elif self.params.action_selector == 2: # epsilon_greedy
            action = self.epsilon_greedy(new_state)
			
        if len(self.memory.memory) > self.params.ER_batch_size and self.params.learning_mode == 1:
            if len(self.memory.memory) > self.params.ER_capacity:
                del self.memory.memory[0]
                
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(self.params.ER_batch_size)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        """Returns sum of rewards"""
        return sum(self.reward_window)/(len(self.reward_window)+1.)
        
    def save_brain(self, path, name):
        """Saving weights from Q-network to the path 'saves/weights' with the specified name specified in parse argument in main"""
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, os.path.join(path, str(name) + '.pth'))
    
    def load_brain(self, path, name):
        """Loading weights from the path 'saves/weights' with the specified name specified in parse argument in main to Q-network"""
        if os.path.isfile(os.path.join(path, str(name) + '.pth')):
            print("=> loading brain... ")
            checkpoint = torch.load(os.path.join(path, str(name) + '.pth'))
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loading is complete !")
        else:
            print("no brain found...")
            exit(1)
            
    def save_experience(self, path, name):
        """Saving experience from experience replay to the path 'saves/experience' with the specified name specified in parse argument in main"""
        with open(path + '/' + name, 'wb') as fp:
            pickle.dump(self.memory, fp)
            
    def load_experience(self, path, name):
        """Loading experience from the path 'saves/experience' with the specified name specified in parse argument in main to experience replay"""
        if os.path.isfile(os.path.join(path, str(name))):
            print("=> loading experience... ")
            with open (path + '/' + name, 'rb') as fp:
                self.memory = pickle.load(fp)
            print("Loading is complete !")
        else:
            print("no experience found...")
            exit(1)