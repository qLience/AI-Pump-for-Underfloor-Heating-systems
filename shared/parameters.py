# This code have been programmed by Christian Blad, Sajuran and Søren Koch aka. Group VT4103A
# The code is based on the original from udemy course https://www.udemy.com/artificial-intelligence-az/
# From Aalborg University

# Gathering all the parameters (that we can modify to explore)
class Params():
    def __init__(self):
    # Parameter comtainer
        self.lr = 0
        self.gamma = 0
        # Softmax
        self.tau = 0
        #Epsilon Greedy
        self.eps_start = 0
        self.eps_end = 0
        self.eps_decay = 0
        # Replay Memory
        self.ER_sample_size = 0
        self.ER_batch_size = 0
        self.ER_capacity = 0
        # Neural Network
        self.input_size = 0
        self.hidden_size = 0
        self.hidden_layers = 0
        self.action_size = 0
        # Eligibility trace steps
        self.n_steps = 0
        # Reference
        self.goalT1 = 0
        self.goalT2 = 0
        self.goalT3 = 0
        self.goalT4 = 0
		# Action selector
        self.action_selector = 1 #1 Softmax #2 Epsilon Greedy