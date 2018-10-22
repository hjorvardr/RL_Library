import numpy as np
import dqn_lib
import ac_lib
import qlearning_lib
import sarsa_lib
from enum import Enum

class EnsemblerType(Enum):
    MAJOR_VOTING_BASED = 0
    AGGREGATION_BASED = 1
    TRUST_BASED = 2



class EnsemblerAgent:
    def __init__(self, output_size, agents, ensembler_type):
        self.agents = agents
        self.output_size = output_size
        self.ensembler_type = ensembler_type

        if self.ensembler_type == EnsemblerType.MAJOR_VOTING_BASED:
            self.votes = np.zeros(self.output_size)
    
    def act(self, state):
        if self.ensembler_type == EnsemblerType.MAJOR_VOTING_BASED:
            for agent in self.agents:
                self.votes[agent.act(state)] += 1
            action = np.argmax(self.votes)
            self.votes = np.zeros(self.output_size)

            return action

        else:
            return 0
