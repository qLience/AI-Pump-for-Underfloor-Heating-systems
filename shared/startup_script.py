# This code have been programmed by Christian Blad, Sajuran and Søren Koch aka. Group VT4103A
# The code is based on the original from udemy course https://www.udemy.com/artificial-intelligence-az/
# From Aalborg University

# Importing libraries
import time

# Environments
SHTL1, SHTL2, SHTL3, SETL1, SETL2, SETL3, ETL1, ETL2, ETL3 = ("shtl1", "shtl2", "shtl3", "setl1", "setl2", "setl3", "etl1", "etl2", "etl3")


class StartUp():
    def __init__(self, params, env, startTmix, env_decider):
        self.params = params
        self.env = env
        self.T1 = 0
        self.T2 = 0
        self.T3 = 0
        self.T4 = 0
        self.Tmix = 0
        self.Treturn = 0
        self.startTmix = startTmix
        self.env_decider = env_decider
        self.WhileHolder = True
        
    def start_script(self):
        """Set environment values to reference temperatures and when rooms are satisfied accordingly to reference temperature then valves is closed to the respective circuit. When the same number of circuits is satisfied accordingly to chosen environment then it will continue to the chosen deep reinforcement learning algorithm."""
        # Set Tmix and open all valves with this initial action
        self.env.sendAction(self.startTmix)
        
        # Up
        while self.WhileHolder:
            print('------------------------------------------------')
            print('Start up script is running')
            # Sleep in order to make sure Simulink and Python can have a solid TCP/IP communication
            time.sleep(0.1)
            #Receive values from Simulink environment
            env_values = self.env.receiveState()
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
               
               
            # Sum rewards and divide by number of circuits
            if self.env_decider == SHTL1 or self.env_decider == SETL1 or self.env_decider == ETL1:
                # Close valves depending on they have achieved reference temperatures
                if self.T1 > self.params.goalT1:
                    self.WhileHolder = False # Continue to DRL framework
                else:
                    self.env.sendAction(self.startTmix) # Satisfy timeout in simulink by keep sending data

            elif self.env_decider == SHTL2 or self.env_decider == SETL2 or self.env_decider == ETL2:
                # Close valves depending on they have achieved reference temperatures
                if self.T1 > self.params.goalT1 and self.T2 > self.params.goalT2:
                    self.env.sendAction(4) # Open both valves
                    self.WhileHolder = False # Continue to DRL framework
                elif self.T1 < self.params.goalT1 and self.T2 < self.params.goalT2:
                    self.env.sendAction(4) # Open both valves
                elif self.T1 > self.params.goalT1:
                    self.env.sendAction(6) # Close valve 1 and open valve 2
                elif self.T2 > self.params.goalT2:
                    self.env.sendAction(5) # Close valve 2 and open valve 1
                else:
                    self.env.sendAction(self.startTmix) # Satisfy timeout in simulink by keep sending data
                    
            elif self.env_decider == SHTL3 or self.env_decider == SETL3 or self.env_decider == ETL3:
                # Close valves depending on they have achieved reference temperatures
                if self.T1 > self.params.goalT1 and self.T2 > self.params.goalT2 and self.T3 > self.params.goalT3 and self.T4 > self.params.goalT4:
                    self.env.sendAction(4) # Open all valves
                    self.WhileHolder = False # Continue to DRL framework
                elif self.T1 > self.params.goalT1 and self.T2 > self.params.goalT2 and self.T3 > self.params.goalT3 and self.T4 > self.params.goalT4:
                    self.env.sendAction(4) # Open all valves
                elif self.T2 > self.params.goalT2 and self.T3 > self.params.goalT3 and self.T4 > self.params.goalT4:
                    self.env.sendAction(5) #1000
                elif self.T1 > self.params.goalT1 and self.T3 > self.params.goalT3 and self.T4 > self.params.goalT4:
                    self.env.sendAction(6) #0100
                elif self.T1 > self.params.goalT1 and self.T2 > self.params.goalT2 and self.T4 > self.params.goalT4:
                    self.env.sendAction(7) #0010
                elif self.T1 > self.params.goalT1 and self.T2 > self.params.goalT2 and self.T3 > self.params.goalT3:
                    self.env.sendAction(8) #0001
                elif self.T3 > self.params.goalT3 and self.T4 > self.params.goalT4:
                    self.env.sendAction(9) #1100
                elif self.T1 > self.params.goalT1 and self.T4 > self.params.goalT4:
                    self.env.sendAction(10) #0110
                elif self.T1 > self.params.goalT1 and self.T2 > self.params.goalT2:
                    self.env.sendAction(11) #0011
                elif self.T2 > self.params.goalT2 and self.T3 > self.params.goalT3:
                    self.env.sendAction(12) #1001
                elif self.T2 > self.params.goalT2 and self.T4 > self.params.goalT4:
                    self.env.sendAction(13) #1010
                elif self.T1 > self.params.goalT1 and self.T3 > self.params.goalT3:
                    self.env.sendAction(14) #0101
                elif self.T4 > self.params.goalT4:
                    self.env.sendAction(15) #1110
                elif self.T3 > self.params.goalT3:
                    self.env.sendAction(16) #1101
                elif self.T2 > self.params.goalT2:
                    self.env.sendAction(17) #1011
                elif self.T1 > self.params.goalT1:
                    self.env.sendAction(18) #0111
                else:
                    self.env.sendAction(self.startTmix) # Satisfy timeout in simulink by keep sending data
            