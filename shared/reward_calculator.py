# This code have been programmed by Christian Blad, Sajuran and Søren Koch aka. Group VT4103A
# From Aalborg University

import numpy as np

# Environments
TL12, TL3, TL4, SETL2, SETL3, SETL4, ETL2, ETL3, ETL4 = ("tl12", "tl3", "tl4", "setl2", "setl3", "setl4", "etl2", "etl3", "etl4")


class RewardCalculator:
    def __init__(self, params, env_decider):
        self.last_distance1 = 0
        self.last_distance2 = 0
        self.last_distance3 = 0
        self.last_distance4 = 0
        self.params = params
        self.env_decider = env_decider
        self.T1 = 0
        self.T2 = 0
        self.T3 = 0
        self.T4 = 0
        self.Tmix = 0
        self.Treturn = 0
        self.count = 0

        
    def calculate_reward(self, env_values, Cn_valves):
        """Returns a reward which is a function of valve position, environment values (temperatures) accordingly to a 
        given reference temperature specified in the parse argument. How the reward is calculated is depending on chosen environment"""
        
        
        # Try, do to that simulink sometimes sends empty arrays
        # This can happen every 2000 or 15000 times
        try:
            # Values from environment
            T1, T2, T3, T4, Tmix, Treturn = env_values[0], env_values[1], env_values[2], env_values[3], env_values[4], env_values[5]
            self.T1, self.T2, self.T3, self.T4, self.Tmix, self.Treturn = T1, T2, T3, T4, Tmix, Treturn
            print('Enrionment Values are [ T1 , T2 , T3 , T4 , Tmix, Treturn ]')
            print('Enrionment Values are [', T1,',', T2,',', T3,',', T4,',', Tmix,',', Treturn,']')
        except:
           T1, T2, T3, T4, Tmix, Treturn = self.T1, self.T2, self.T3, self.T4, self.Tmix, self.Treturn
           self.count += 1
           
        print('except  called ', self.count)
		# Absolute distance from temperatures to goal
        distance1 = abs(self.params.goalT1 - T1)
        distance2 = abs(self.params.goalT2 - T2)
        distance3 = abs(self.params.goalT3 - T3)
        distance4 = abs(self.params.goalT4 - T4)
        print('distance1 is ', distance1)
        print('distance2 is ', distance2)
        print('distance3 is ', distance3)
        print('distance4 is ', distance4)
        
        # Define room and Tmix temperature limits to environments
        if self.env_decider == TL12 or self.env_decider == TL3 or self.env_decider ==TL4:
            Room_LL = 15.1 # lower limit
            Room_UL = 29.9 # Upper limit
            Tmix_LL = 15.1 # Lower limit
            Tmix_UL = 44.9 # Upper limit
        else: # Experiment
            Room_LL = 24.1 # lower limit
            Room_UL = 34.9 # Upper limit
            Tmix_LL = 24.1 # Lower limit
            Tmix_UL = 59.9 # Upper limit
        
        # Valve in circuit one is always true when running TL12 and ETL2
        if self.env_decider == TL12 or self.env_decider == ETL2  or self.env_decider == SETL2:
            Cn_valves.C1_valve = True
        
        
        # Allowed distance from reference room temperature
        max_dist = 0.5
        
		# Reward Policy - Circuit 1
        if 0 <= distance1 <= max_dist and Cn_valves.C1_valve:
            last_reward1 = 1
            if distance1 < self.last_distance1:
                last_reward1 = last_reward1* (1 - distance1)
        elif 0 <= distance1 <= max_dist:
            last_reward1 = 0.5
            if distance1 < self.last_distance1:
                last_reward1 = last_reward1* (1 - distance1)
        elif distance1 < self.last_distance1:
            last_reward1 = 0.1
        elif T1 < Room_LL or T1 > Room_UL or Tmix < Tmix_LL or Tmix > Tmix_UL:
            last_reward1 = -1
        else:
            last_reward1 = -0.1*distance1
            
            
		# Reward Policy - Circuit 2
        if 0 <= distance2 <= max_dist and Cn_valves.C2_valve:
            last_reward2 = 1
            if distance2 < self.last_distance2:
                last_reward2 = last_reward2* (1 - distance2)
        elif 0 <= distance2 <= max_dist:
            last_reward2 = 0.5
            if distance2 < self.last_distance2:
                last_reward2 = last_reward2* (1 - distance2)
        elif distance2 < self.last_distance2:
            last_reward2 = 0.1
        elif T2 < Room_LL or T2 > Room_UL or Tmix < Tmix_LL or Tmix > Tmix_UL:
            last_reward2 = -1
        else:
            last_reward2 = -0.1*distance2
            
            
		# Reward Policy - Circuit 3
        if 0 <= distance3 <= max_dist and Cn_valves.C3_valve:
            last_reward3 = 1
            if distance3 < self.last_distance3:
                last_reward3 = last_reward3* (1 - distance3)
        elif 0 <= distance3 <= max_dist:
            last_reward3 = 0.5
            if distance3 < self.last_distance3:
                last_reward3 = last_reward3* (1 - distance3)
        elif distance3 < self.last_distance3:
            last_reward3 = 0.1
        elif T3 < Room_LL or T3 > Room_UL or Tmix < Tmix_LL or Tmix > Tmix_UL:
            last_reward3 = -1
        else:
            last_reward3 = -0.1*distance3
            
            
		# Reward Policy - Circuit 4
        if 0 <= distance4 <= max_dist and Cn_valves.C4_valve:
            last_reward4 = 1
            if distance4 < self.last_distance4:
                last_reward4 = last_reward4* (1 - distance4)
        elif 0 <= distance4 <= max_dist:
            last_reward4 = 0.5
            if distance4 < self.last_distance4:
                last_reward4 = last_reward4* (1 - distance4)
        elif distance4 < self.last_distance4:
            last_reward4 = 0.1
        elif T4 < Room_LL or T4 > Room_UL or Tmix < Tmix_LL or Tmix > Tmix_UL:
            last_reward4 = -1
        else:
            last_reward4 = -0.1*distance4
            
        #Update
        self.last_distance1 = distance1
        self.last_distance2 = distance2
        self.last_distance3 = distance3
        self.last_distance4 = distance4
        
        # Sum rewards and divide by number of circuits
        if self.env_decider == TL12 or self.env_decider == ETL2 or self.env_decider == SETL2:
            last_reward = (last_reward1) / 1
        elif self.env_decider == TL3 or self.env_decider == ETL3 or self.env_decider == SETL3:
            last_reward = (last_reward1 + last_reward2) / 2
        elif self.env_decider == TL4 or self.env_decider == ETL4  or self.env_decider == SETL4:
            last_reward = (last_reward1 + last_reward2 + last_reward3 + last_reward4) / 4

        
        return last_reward
