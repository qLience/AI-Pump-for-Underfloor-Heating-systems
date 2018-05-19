# This code have been programmed by Christian Blad, Sajuran and Søren Koch aka. Group VT4103A
# The code is based on the original from udemy course https://www.udemy.com/artificial-intelligence-az/
# From Aalborg University

# Importing the libraries
import matplotlib.pyplot as plt
import os
import time
from matplotlib.backends.backend_pdf import PdfPages

# Creating the
class Training():
    def __init__(self, params, model_type, env, reward_calculator, ai_input_provider):
        self.brain = model_type
        self.env = env
        self.env_values = []
        self.reward_calculator = reward_calculator
        self.ai_input_provider = ai_input_provider
        self.last_reward = 0
        self.scores = []
        self.action = 0

    def update(self):
        """This function update the chosen deep reinforcement learning framework with states 
        and the environment with actions"""
        # Sleep in order to make sure Simulink and Python can have a solid TCP/IP communication
        time.sleep(0.1)
        
        #Receive values from Simulink environment
        self.env_values = self.env.receiveState()
        
        # Convert environment values to states 
        state = self.ai_input_provider.calculate_ai_input(self.env_values, self.action)
        print('State inputs to brain')
        print(state)
        
        # Update brain
        action = self.brain.update(self.last_reward, state)
        
        # Send action to agent in environment
        self.env.sendAction(action + 1)
        print('action is ', action + 1)
        
        # Calculate reward from environment values
        self.last_reward = self.reward_calculator.calculate_reward(self.env_values, self.ai_input_provider)
        print('reward is ', self.last_reward)
        
        #update score
        self.scores.append(self.brain.score())
        
        #update action
        self.action = action + 1
        
    
    def actionFromNN(self):
        """Returns action from Q-network"""
        return self.action
	
    def getScores(self):
        """Returns the agents' score"""
        return scores
		
	# Saving experience
    def save(self, path, name):
        """Saving training curves tracking the agent score to the path 'saves/plot' with the specified name specified in parse argument in main"""
        plt.plot(self.scores, color='red')
        plt.ylabel('reward score per iterations')
        plt.xlabel('Training iterations')
        plt.title('Training curves tracking the agent score')
        plt.savefig(os.path.join(path, str(name) + '.pdf'), format='pdf')