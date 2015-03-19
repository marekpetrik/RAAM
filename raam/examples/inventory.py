"""
=============================================================================
Energy arbitrage with degradable storage (:mod:`raam.examples.inventory`)
=============================================================================

Models an invetory problem with multiple price levels, sales, purchases. The storage
is assumed to degrade with use. This can be used to model battery energy storage
with the battery degrading while in use.
"""
import numpy as np
import scipy as sp
import itertools
import raam
import pickle
from raam import features

def degradation(fun,charge,discharge):
    """
    Constructs a capacity degradation function from given parameters.
    
    The functions are D- and D+: the antiderivatives. See the paper for more details
    
    Parameters
    ----------
    fun : {"polynomial"}
        Type of the function
    charge : object
        List of parameters for the function. 
        polynomial: (charge,discharge)
            parameters are the coefficients with low to high order    
    discharge : object
        List of parameters for the function. 
        polynomial: (charge,discharge)
            parameters are the coefficients with low to high order    

    Returns
    -------
    out : function (x1,x2) 
        x1: initial level
        x2: new level
    """
    if fun == 'polynomial':
        charge_a = np.array(charge)
        discharge_a = np.array(discharge)
        def polynomial(x1,x2):
            if x1 <= x2:    # charge
                # intentional x2 - x1
                return np.polynomial.polynomial.polyval(x2,charge_a) - np.polynomial.polynomial.polyval(x1,charge_a) 
            else:           # discharge
                # intentional: x1 - x2
                return np.polynomial.polynomial.polyval(x1,discharge_a) - np.polynomial.polynomial.polyval(x2,discharge_a) 
        return polynomial
    else:
        raise ValueError('Incorrect function type: "%s"' % funtype)

DefaultConfiguration = {
    "price_buy" : [1.2,2.1,3.3],
    "price_sell" : [1,2,3],
    "price_probabilities" : np.array([[0.8, 0.1, 0.1],[0.1, 0.8, 0.1],[0.1, 0.1, 0.8]]),
    "initial_capacity" : 1,
    "initial_inventory" : 0.5,
    "degradation" : {"fun":"polynomial","charge":[0,0,0.01],"discharge":[0.01,-0.02,0.01]},
    "capacity_cost" : 1,
    "change_capacity" : True
    }


class Simulator(raam.Simulator):
    """
    Simulates the evolution of the inventory, the total capacity, and the price 
    levels. The prices come from a Markov chain process.
    
    The initial state is generated from an expectation state 
    
    Decision state (tuple): 
        - inventory
        - capacity
        - priceindex 
    
    Expectation state: 
        - inventory
        - capacity
        - priceindex
        - reward
    
    Action: charge change

    Parameters
    ----------

    """

    def __init__(self,config,discount=0.9999):
        """ 
        If no filename and no configuration are provided, 
        then the configuration is created automatically 
        """
        self.discount = discount
        self.degradation = degradation(**config['degradation'])
        self.initial_capacity = config['initial_capacity']
        self.price_buy = config['price_buy']
        self.price_sell = config['price_sell']
        self.price_probabilities = config['price_probabilities']
        self.capacity_cost = config['capacity_cost']
        self.change_capacity = config['change_capacity']
        self.initial_inventory = config['initial_inventory']

        assert np.all(np.array(self.price_buy) >= 0)
        assert np.all(np.array(self.price_sell) >= 0)
        assert len(self.price_buy) == len(self.price_sell)
        assert self.price_probabilities.shape[0] == self.price_probabilities.shape[1] == len(self.price_sell)
        assert np.max(np.abs(np.sum(self.price_probabilities,1) - 1)) < 0.01
        assert np.all(np.array(self.price_probabilities) >= 0)
        assert self.capacity_cost >= 0
        assert type(self.change_capacity) is bool
        assert self.initial_inventory <= self.initial_capacity and self.initial_inventory >= 0 

    @property
    def discount(self):
        return self.discount

    def transition_dec(self,decstate,action):
        """ 
        Represents a transition from the deterministic state to an expectation state.

        Parameters
        ----------
        decstate : decision state
            inventory,capacity,pricebuy,pricesell 
        action : float
            change in charge

        Returns
        -------
        out : expectation state
            inventory,capacity,reward
        """
        inventory, capacity, pricebuy, pricesell = decstate

        assert inventory >= 0
        assert inventory <= capacity

        action = max(action, - inventory)
        action = min(action, capacity - inventory)

        ninventory = inventory + action
        capacity_loss = self.degradation(ninventory / capacity, inventory / capacity) * capacity
        assert capacity_loss >= -1e-10
        if self.change_capacity:
            ncapacity = max(0,capacity - capacity_loss)
            ninventory = min(ninventory, ncapacity)
        else:
            ncapacity = capacity

        reward = (pricebuy if action >= 0 else pricesell) * action
        reward -= capacity_loss * self.capacity_cost
        return (ninventory,ncapacity,reward)

    def transition_exp(self,expstate):
        """ 
        Transition from an expectation state to a decision state

        Parameters
        ----------
        expstate : expectation state

        Returns
        -------
        reward : float
        decision state : object
        """
        inventory,capacity,reward = expstate
        
        assert inventory >= 0
        assert inventory <= capacity

        priceindex = np.random.choice(np.array(range(len(self.price_probabilities))), p=self.price_probabilities)
        return (reward,(inventory,capacity,self.price_buy[priceindex],self.price_sell[priceindex]))

    def actions(self,decstate):
        inventory, capacity, _, _ = decstate
        return np.arange(-inventory, capacity - inventory, 0.05)

    def stateiterator(self):
        expstate = (self.initial_inventory,self.initial_capacity,0)
        while True:
            yield self.transition_exp(expstate)[1]

    def initstates(self):
        return self.stateiterator()

class Features:
    """ 
    Suitable features for inventory management 
    """
    linear = (features.piecewise_linear(None), features.piecewise_linear(None))

