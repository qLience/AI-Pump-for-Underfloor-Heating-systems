# This code have been programmed by Christian Blad, Sajuran and Søren Koch aka. Group VT4103A
# The code is based on the original from udemy course https://www.udemy.com/artificial-intelligence-az/
# From Aalborg University


# Environments
SHTL1, SHTL2, SHTL3, SETL1, SETL2, SETL3, ETL1, ETL2, ETL3 = ("shtl1", "shtl2", "shtl3", "setl1", "setl2", "setl3", "etl1", "etl2", "etl3")


class AiInputProvider:
    def __init__(self, params, env_decider):
        self.params = params
        self.env_decider = env_decider
        self.T1 = 0
        self.T2 = 0
        self.T3 = 0
        self.T4 = 0
        self.Tmix = 0
        self.Treturn = 0
        self.last_T1 = 0
        self.last_T2 = 0
        self.last_T3 = 0
        self.last_T4 = 0
        self.C1_valve = 0
        self.C2_valve = 0
        self.C3_valve = 0
        self.C4_valve = 0
        
    def calculate_ai_input(self, env_values, action):
        """Returns a compress state vector depending on chosen environment.
        This function standardise environment values so it become state inputs to the Q-network."""
        # Try, do to that simulink sometimes sends empty arrays
        # This can happen every 2000 or 15000 times
        try:
            # Values from environment
            T1, T2, T3, T4, Tmix, Treturn = env_values[0], env_values[1], env_values[2], env_values[3], env_values[4], env_values[5]
            self.T1, self.T2, self.T3, self.T4, self.Tmix, self.Treturn = T1, T2, T3, T4, Tmix, Treturn
        except:
           T1, T2, T3, T4, Tmix, Treturn = self.T1, self.T2, self.T3, self.T4, self.Tmix, self.Treturn
        
        # Standadize input data
        # Orientation
        if (T1-self.params.goalT1) <= 0:
            orientation1_std = 0.5
        else:
            orientation1_std = -0.5

        if (T2-self.params.goalT2) <= 0:
            orientation2_std = 0.5
        else:
            orientation2_std = -0.5
            
        if (T3-self.params.goalT3) <= 0:
            orientation3_std = 0.5
        else:
            orientation3_std = -0.5

        if (T4-self.params.goalT4) <= 0:
            orientation4_std = 0.5
        else:
            orientation4_std = -0.5
            
        # Room Temperature
        T1_std = (T1 ) / 35
        T2_std = (T2 ) / 35
        T3_std = (T3 ) / 35
        T4_std = (T4 ) / 35
        
        # Diff
        diff1_std = abs((T1 - self.last_T1)* 10)
        diff2_std = abs((T2 - self.last_T2)* 10)
        diff3_std = abs((T3 - self.last_T3)* 10)
        diff4_std = abs((T4 - self.last_T4)* 10)
        
        # Update
        self.last_T1 = T1
        self.last_T2 = T2
        self.last_T3 = T3
        self.last_T4 = T4
        
        orientation1 = (T1-self.params.goalT1)
        orientation2 = (T2-self.params.goalT2)
        orientation3 = (T3-self.params.goalT3)
        orientation4 = (T4-self.params.goalT4)
        
        diff1 = abs(T1 - self.last_T1)
        diff2 = abs(T2 - self.last_T2)
        diff3 = abs(T3 - self.last_T3)
        diff4 = abs(T4 - self.last_T4)
        
        # Standadize mixing temperature to environment
        if self.env_decider == SHTL1 or self.env_decider == SHTL2 or self.env_decider == SHTL3:
            Tmix_std = (Tmix - 15)/30
        else:
            Tmix_std = (Tmix - 15)/45
        
        # Compress state vector to specific environment
        if self.env_decider == SHTL1 or self.env_decider == SETL1 or self.env_decider == ETL1:
            # Compress to state vector
            state =  [T1_std, orientation1_std, diff1_std, Tmix_std]
            
        elif self.env_decider == SHTL2 or self.env_decider == SETL2 or self.env_decider == ETL2:
            # Circuit valves open or close ?
            if action ==  4:
                self.C1_valve = 1
                self.C2_valve = 1
            elif action == 5:
                self.C1_valve = 1
                self.C2_valve = 0
            elif action == 6:
                self.C1_valve = 0
                self.C2_valve = 1
            elif action == 7:
                self.C1_valve = 0
                self.C2_valve = 0
            # Compress to state vector
            state =  [T1_std, orientation1_std, diff1_std, self.C1_valve, T2_std, orientation2_std, diff2_std, self.C2_valve, Tmix_std]
        
        elif self.env_decider == SHTL3 or self.env_decider == SETL3 or self.env_decider == ETL3:
            # Circuit valves open or close ?
            if action ==  4:
                self.C1_valve = 1
                self.C2_valve = 1
                self.C3_valve = 1
                self.C4_valve = 1
            elif action ==  5:
                self.C1_valve = 1
                self.C2_valve = 0
                self.C3_valve = 0
                self.C4_valve = 0
            elif action ==  6:
                self.C1_valve = 0
                self.C2_valve = 1
                self.C3_valve = 0
                self.C4_valve = 0
            elif action ==  7:
                self.C1_valve = 0
                self.C2_valve = 0
                self.C3_valve = 1
                self.C4_valve = 0
            elif action ==  8:
                self.C1_valve = 0
                self.C2_valve = 0
                self.C3_valve = 0
                self.C4_valve = 1
            elif action ==  9:
                self.C1_valve = 1
                self.C2_valve = 1
                self.C3_valve = 0
                self.C4_valve = 0
            elif action ==  10:
                self.C1_valve = 0
                self.C2_valve = 1
                self.C3_valve = 1
                self.C4_valve = 0
            elif action ==  11:
                self.C1_valve = 0
                self.C2_valve = 0
                self.C3_valve = 1
                self.C4_valve = 1
            elif action ==  12:
                self.C1_valve = 1
                self.C2_valve = 0
                self.C3_valve = 0
                self.C4_valve = 1
            elif action ==  13:
                self.C1_valve = 1
                self.C2_valve = 0
                self.C3_valve = 1
                self.C4_valve = 0
            elif action ==  14:
                self.C1_valve = 0
                self.C2_valve = 1
                self.C3_valve = 0
                self.C4_valve = 1
            elif action ==  15:
                self.C1_valve = 1
                self.C2_valve = 1
                self.C3_valve = 1
                self.C4_valve = 0
            elif action ==  16:
                self.C1_valve = 1
                self.C2_valve = 1
                self.C3_valve = 0
                self.C4_valve = 1
            elif action ==  17:
                self.C1_valve = 1
                self.C2_valve = 0
                self.C3_valve = 1
                self.C4_valve = 1
            elif action ==  18:
                self.C1_valve = 0
                self.C2_valve = 1
                self.C3_valve = 1
                self.C4_valve = 1
            elif action ==  19:
                self.C1_valve = 0
                self.C2_valve = 0
                self.C3_valve = 0
                self.C4_valve = 0
            # Compress to state vector
            state =  [T1_std, orientation1_std, diff1_std, self.C1_valve, T2_std, orientation2_std, diff2_std, self.C2_valve, T3_std, orientation3_std, diff3_std, self.C3_valve, T4_std, orientation4_std, diff4_std, self.C4_valve, Tmix_std]
        
        return state
        