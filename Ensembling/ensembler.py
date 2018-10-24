import numpy as np
import dqn_lib
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
        
        if self.ensembler_type == EnsemblerType.TRUST_BASED:
            self.votes = np.zeros(len(self.agents))
            self.probs = np.zeros(self.output_size)
            self.trust = np.zeros(len(self.agents))
            self.trust_rate = 0.1

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
        
        if self.ensembler_type == EnsemblerType.AGGREGATION_BASED:
            for agent in self.agents:
                suggested_action, probs = agent.act(state) # TODO: check probs
                #probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))
                self.probs += probs
            self.probs = self.probs / np.sum(self.probs)
            #print(self.probs)
            action = np.argmax(self.probs)
            self.probs = np.zeros(self.output_size)
            return action
            
        if self.ensembler_type == EnsemblerType.TRUST_BASED:
            for i in range(len(self.agents)):
                agent = self.agents[i]
                suggested_action, probs = agent.act(state)
                #probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))
                self.probs += probs * self.trust[i]

            self.probs = self.probs / np.sum(self.probs)

            action = np.argmax(self.probs)
            
            self.probs = np.zeros(self.output_size)
            
            for i in range(len(self.agents)):   
                agent = self.agents[i]
                suggested_action, probs = agent.act(state)
                if action == suggested_action:
                    self.votes[i] += 1
            return action    

        return 73 # Huston, we have a problem!
    
    def trust_update(self, end, score):
        if self.ensembler_type == EnsemblerType.TRUST_BASED:
            if end:
                print(self.votes, "score:", score)
                for i in range(len(self.agents)):
                    self.trust[i] = self.trust[i] * ((1 - score) + (self.votes[i] / sum(self.votes)))
                
                self.trust = self.trust / sum(self.trust)
                
                self.votes = np.zeros(len(self.agents))
                
                print(self.trust)
                    
