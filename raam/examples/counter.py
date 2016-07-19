"""
=================================================
Simple test simulator (raam.examples.counter)
=================================================

Counts up and down with the actions
"""
import random
import itertools
import raam

class Counter(raam.simulator.Simulator):  
    """
    Decision state: position in chain
    Initial (decision) state: 0
    Actions: {plus, minus}
    Rewards: succ_prob%: next position, 1-succ_prob% this position in chain

    Parameters
    ----------
    succ_prob : float
        Success probability of the action taken; otherwise the transition does
        not happen.
    """
    def __init__(self,succ_prob=0.9):
        self.succ_prob = succ_prob

    @property
    def discount(self):
        return 0.9
        
    def transition(self,pos,action):
       
        if random.random() <= self.succ_prob:
            if action == 'plus':
                act = +1
            elif action == 'minus':
                act = -1
            else:
                raise ValueError('Invalid action')

            return pos,pos + act
        else:
            return pos,pos
            
    def end_condition(self,decstate):
        return False
        
    def initstates(self):
        return itertools.cycle([0])
        
    def actions(self,decstate):
        return ['plus','minus']

