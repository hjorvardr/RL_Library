import numpy as np
import dqn_lib
from enum import Enum

class EnsemblerType(Enum):
    MAJOR_VOTING_BASED = 0
    TRUST_BASED = 1


class EnsemblerAgent:
    def __init__(self, output_size, agents, ensembler_type):
        self.agents = agents
        self.output_size = output_size
        self.ensembler_type = ensembler_type

        if self.ensembler_type == EnsemblerType.MAJOR_VOTING_BASED:
            self.votes = np.zeros(self.output_size)
        
        if self.ensembler_type == EnsemblerType.AGGREGATION_BASED:
            self.probs = np.zeros(self.output_size)
        
        if self.ensembler_type == EnsemblerType.TRUST_BASED:
            self.votes_per_agent = np.zeros(len(self.agents))
            self.votes = np.zeros(self.output_size)
            self.trust = np.zeros(len(self.agents))
            self.trust_rate = 0.1
            self.total_actions = 0

            for i in range(len(self.trust)):
                self.trust[i] = 1 / len(self.agents)
            print("INITIAL TRUST: ", self.trust)

    def act(self, state):
        if self.ensembler_type == EnsemblerType.MAJOR_VOTING_BASED:
            for agent in self.agents:
                suggested_action, probs = agent.act(state)
                self.votes[suggested_action] += 1
            action = np.argmax(self.votes)
            self.votes = np.zeros(self.output_size)
            return action
        
        if self.ensembler_type == EnsemblerType.TRUST_BASED:
            for i in range(len(self.agents)):
                agent = self.agents[i]
                suggested_action, probs = agent.act(state)
                self.votes[suggested_action] += self.trust[i]

            action = np.argmax(self.votes)
            
            self.votes = np.zeros(self.output_size)
            
            for i in range(len(self.agents)):   
                agent = self.agents[i]
                suggested_action, probs = agent.act(state)
                if action == suggested_action:
                    self.votes_per_agent[i] += 1
            self.total_actions += 1
            return action    

        return 73 # Houston, we have a problem!
    
    def trust_update(self, win):
        if self.ensembler_type == EnsemblerType.TRUST_BASED:
            print(self.votes_per_agent, "win:", win, "total actions:", self.total_actions)
            for i in range(len(self.agents)):
                if win:
                    self.trust[i] = self.trust[i] * (1 + self.trust_rate * (self.votes_per_agent[i] / self.total_actions))
                else:
                    self.trust[i] = self.trust[i] * (1 - self.trust_rate * (self.votes_per_agent[i] / self.total_actions))
            
            self.trust = self.trust / sum(self.trust)
            self.votes_per_agent = np.zeros(len(self.agents))
            self.total_actions = 0
            print(self.trust)
                    
