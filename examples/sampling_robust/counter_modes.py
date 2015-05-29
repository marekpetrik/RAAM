"""
A simple example adapted from ``raam.examples.counter``
"""
import random
import itertools
import raam
import numpy as np

class CounterModes(raam.simulator.Simulator):  
    """
    A simple chain counter class with different modes. The modes differ in their 
    transition probabilties. Modes can be changed but also may change randomly
    
    If the left-right action fails, then the next state is chosen according to 
    a randomly generated transition probability. The transition probability increases
    
    
    Decision state: position in chain, mode
    
    Expectation state: decision state, action
    
    Initial (decision) state: 0
    
    Actions: {plus, minus, modeup, modedown}
    
    Rewards: 
        __succ_prob: next position, 
        1-__succ_prob this position in chain

    Parameters
    ----------
    rewards : vector
        Vector of rewards, also determines the number of states
    modecount : int
        Number of modes. The reward in all modes are the same. The transition 
        probabilities may differ.
    succ_prob : float
        Probabilities that the left-right actions succeed, otherwise the move is 
        with probabilities given by an initally constructed random distribution.
        
    """
    def __init__(self,rewards,modecount,succ_prob):
        self.__statecount = len(rewards)
        self.__rewards = rewards
        self.__modecount = modecount
        self.__succ_prob = succ_prob

        # the probability that the mode changes when no mode change action is taken
        self.__mode_change_prob = 0.7

        # the acceptable modes are used to compute the baseline policy,
        # which should be optimal for mode 0
        self.__acceptablemodecount = self.__modecount 

        # construct the base transition for mode 0
        P0 = np.array([np.random.dirichlet(np.ones(self.__statecount))  
                            for i in range(self.__statecount)])
        
        P1 = np.array([np.random.dirichlet(np.ones(self.__statecount))  
                            for i in range(self.__statecount)])
        
        self.__P = []
        for m in range(self.__modecount):
            c = m / self.__modecount            
            self.__P.append((1-c) * P0 + c * P1)

    @property
    def discount(self):
        return 0.9
        
    def set_acceptable_modes(self,newmodecount):
        """
        Increases or decreases the number of acceptable modes to ``newmodecount''
        """
        assert newmodecount >= 0 and newmodecount <= self.__modecount
        self.__acceptablemodecount = newmodecount
        
    def transition_dec(self,decstate,action):
        return (decstate, action)
        
    def transition_exp(self,expstate):
        (pos,mode),action = expstate

        # npos - new position
        # nmode - new mode
        
        # update the new position
        if random.random() <= self.__succ_prob:
            # make the change according to the success probability
            if action == 'plus':
                npos = pos + 1
            elif action == 'minus':
                npos = pos - 1
            else:
                npos = pos            
            
            npos = min(max(0, npos),self.__statecount-1)
        else:
            # make the transition randomly according to the change probability
            
            p = self.__P[pos,:]
            npos = np.random.choice(np.arange(self.__statecount), p=p)
            
        # update the new mode
        if action == 'modeup':
            nmode = mode + 1
        elif action == 'modedown':
            nmode = mode - 1
        else:
            q = random.random()
            if q < self.__mode_change_prob/2:
                nmode = mode + 1
            elif q < self.__mode_change_prob:
                nmode = mode - 1
                
        nmode = min(max(nmode,0),self.__acceptablemodecount-1)
        
        return (npos,nmode), self.__rewards[pos]
            
    def end_condition(self,decstate):
        return False
        
    def initstates(self):
        return itertools.cycle( (0,0) )
        
    def actions(self,decstate):
        return ['plus','minus','modeup','modedown']

