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
        if self.ensembler_type == EnsemblerType.AGGREGATION_BASED:
            self.probs = np.zeros(self.output_size)
        if self.ensembler_type == EnsemblerType.AGGREGATION_BASED:
            self.votes = np.zeros(self.output_size)
            self.probs = np.zeros(self.output_size)
            self.trust = np.zeros(len(self.agents))
            for i in range(len(self.trust)):
                self.trust[i] = 1 / len(self.agents))
    
    def act(self, state):
        if self.ensembler_type == EnsemblerType.MAJOR_VOTING_BASED:
            for agent in self.agents:
                self.votes[agent.act(state)] += 1
            action = np.argmax(self.votes)
            self.votes = np.zeros(self.output_size)
            return action
        
        if self.ensembler_type == EnsemblerType.AGGREGATION_BASED:
            for agent in self.agents:
                suggested_action, probs = agent.act(state)
                self.probs += probs
            action = np.argmax(self.probs)
            self.probs = np.zeros(self.output_size)
            return action
            
        if self.ensembler_type == EnsemblerType.TRUST_BASED:
            for i in range(len(self.agents)):
                agent = agents[i]
                suggested_action, probs = agent.act(state)
                self.votes += probs * self.trust[i]
            action = np.argmax(self.votes)
            
            for agent in self.agents:
                suggested_action, probs = agent.act(state)
                if action == suggested_action:
                    self.votes[suggested_action] += 1
            return action    
            
        return 0
    
    def trust_update(self, end, score):
        if self.ensembler_type == EnsemblerType.TRUST_BASED:
            if end:
                if score > 0:
                    reward = 0.25 
                else:
                    reward = -0.25
                for i in range(len(self.agents)):
                    self.trust[i] += reward / sum(self.votes) * self.votes[i]
                self.trust = self.trust / sum(self.trust)
                    
