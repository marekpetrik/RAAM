"""
===============================================================
Inverted pendulum balancing (:mod:`raam.examples.pendulum`)
===============================================================
    
An inverted pendulum (http://en.wikipedia.org/wiki/Inverted_pendulum) 
is a pendulum which has its center of mass above its pivot point. 
It is often implemented with the pivot point mounted on a cart that can move horizontally 
and may be called a cart and pole.
"""
import raam
import numpy as np
import math
import itertools
from raam import features

class Simulator(raam.Simulator):  
    """
    Simulator of the inverted pendulum:
        * Decision state: decstate = (theta, dtheta)
    """

    actionset = [-50,0,50]

    def __init__(self,g=9.8,m=2.0,M=8.0,l=0.5):
        """
        Initializes the pendulum class 

        Parameters
        ----------
        g : float, optional
            Gravitational coefficient
        m : float, optional
            Mass on the top of the pendulum
        M : float, optional
            Mass of the cart
        l : float, optional
            Length of the inverted pendulum
        """
        self.g = 9.8
        self.m = 2.0
        self.M = 8.0
        self.l = 0.5
        
    @property
    def discount(self):
        return 0.99

    def transition(self,decstate,action):
        """ expstate = (decstate, action) """
        theta = decstate[0] 
        dtheta = decstate[1]
        alpha = 1/(self.m + self.M)
            
        u = action + np.random.uniform(-10,10)
        ddtheta = (self.g * math.sin(theta) - \
                   alpha*self.m*self.l*(dtheta**2)*math.sin(2*theta)/2 - \
                   alpha*math.cos(theta)*u) / \
                   (4*self.l/3 - alpha*self.m*self.l*(math.cos(theta))**2)
            
        dtheta = dtheta + ddtheta*0.1
        theta = theta + dtheta*0.1

        reward = -1*int(abs(theta) >= math.pi/2)
            
        return reward,[theta,dtheta]

    def initial_state_sample(self):
        """" Return a sampled initial state """
        return [np.random.uniform(-math.pi/8,math.pi/8), np.random.uniform(-1/2,1/2)]

    def actions(self,decstate):
        return Simulator.actionset

    def initstates(self):
        return ( self.initial_state_sample() for _ in itertools.count() )

    def end_condition(self,decstate):
        return abs(decstate[0]) > math.pi/2 
        
    def policy_left(self):
        """ Returns a policy that always takes the left-most action"""
        assert len(self.actionset) == 3
        return lambda x: self.actionset[0]

    def policy_neutral(self):
        """ Returns a policy that always takes the neutral action"""
        assert len(self.actionset) == 3
        return lambda x: self.actionset[1]
    
    def policy_right(self):
        """ Returns a policy that always takes the right-most action"""
        assert len(self.actionset) == 3
        return lambda x: self.actionset[2]

class Simulator2(Simulator):
    """ The same pendulum simulator, but it does not stop """

    def end_condition(self,decstate):
        return False

# Representation of the decision and expectation state

def decstate_rep(decstate):
    """
    List representation of a decision state.
    """
    theta = decstate[0] 
    dtheta = decstate[1]
    return [theta, dtheta] 


class Features:
    """ 
    Features used in the approximation of the pendulum value function 
    (decision state function, expectation state function)
    """
    points = list(itertools.product( \
                np.linspace(-math.pi/4, math.pi/4, num=3), \
                np.linspace(-1, 1, num=3)))

    gaussian = (features.gaussian(points), features.gaussian_action(points,Simulator.actionset) )
