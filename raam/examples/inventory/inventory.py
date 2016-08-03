import numpy as np
import itertools
import raam
import pickle
from raam import features
from scipy import cluster

def degradation(fun,charge,discharge):
    """
    Constructs a capacity degradation function from given parameters.
    
    The functions are D- and D+: the anti-derivatives. See the paper for more details
    
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
    "degradation" : {"fun":"polynomial","charge":[0,0,0.01],"discharge":[0.01,0.02,0.01]},
    "capacity_cost" : 1,
    "change_capacity" : True,
    "charge_limit" : 1, # limit on charge in a single step
    "discharge_limit" : 1  # limit on discharge in a single step, absolute value
    }


class Simulator(raam.Simulator):
    """
    Simulates the evolution of the inventory, the total capacity, and the price 
    levels. The prices come from a Markov chain process.
    
    The initial state is generated from an expectation state 
    
    State (tuple): 
        - inventory
        - capacity
        - priceindex 
    
    Action: charge change
        This is not an index but an absolute value of the change in charge

    Parameters
    ----------
    config : dict
        Configuration. See DefaultConfiguration for an example
    discount : float, optional
        Discount factor
    action_cnt : int, optional
        Number of ticks for discretizing actions.
    inventory_cnt : int, optional
        Number of ticks for discretizing inventory states. 
    capacity_cnt : int, optional
        Discretization set for storage capacity states. 
        This step must be fine enough to capture the small
        change in capacities
    """

    def __init__(self,config,discount=0.9999,action_cnt=20,inventory_cnt=100,\
                    capacity_cnt=100):
        self._discount = discount
        self._action_cnt = action_cnt
        self._inventory_cnt = inventory_cnt
        self._capacity_cnt = capacity_cnt
        
        self.degradation = degradation(**config['degradation'])
        self.initial_capacity = config['initial_capacity']
        self.price_buy = config['price_buy']
        self.price_sell = config['price_sell']
        self.price_probabilities = config['price_probabilities']
        self.capacity_cost = config['capacity_cost']
        self.change_capacity = config['change_capacity']
        self.initial_inventory = config['initial_inventory']

        if 'charge_limit' not in config:
            warn('No charge_limit in config, using 1')
        if 'discharge_limit' not in config:
            warn('No disccharge_limit in config, using 1')
        self.charge_limit = config['charge_limit'] if 'charge_limit' in config else 1
        self.discharge_limit = config['discharge_limit'] if 'discharge_limit' in config else 1

        # state and the distributions
        self._all_states = None
        self._initial_distribution = None

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
        return self._discount

    def get_stateindex(self, decstate):
        """
        Finds the index of the state in the list returned by all_states
        """
        # lazy initialization
        if self._all_states is None:
            self.all_states()

        return self._state_aggregation.classify(decstate)


    def all_states(self):
        """
        Returns all states (quantized according to the parameters provided in the constructor)

        There is no iteration over capacities if self.change_capacity = False and it is 
        fixed to be 0.

        Important: Use self.get_stateindex() to get index of a state instead of searching this list.
        It is much more efficient.

        Returns
        -------
        out : np.ndarray
            List of all states
        """
        
        # lazy initialization
        if self._all_states is None:

            pricecount = len(self.price_buy)

            # if the capacity does not change, then aggregate the capacity dimension to only one value
            if self.change_capacity:
                # make sure that the centers of price clusters are integer numbers
                self._state_aggregation = raam.features.GridAggregation(\
                         ((0,self.initial_capacity), (0,self.initial_capacity), (-0.5,pricecount-0.5)), \
                         (self._inventory_cnt, self._capacity_cnt, pricecount)  )
            else:
                self._state_aggregation = raam.features.GridAggregation(\
                         ((0,self.initial_capacity), (self.initial_capacity-0.1,self.initial_capacity+0.1), (-0.5,pricecount-0.5)), \
                         (self._inventory_cnt, 1, pricecount)  )

            self._all_states = [s for s in self._state_aggregation.centers() if s[0] <= s[1]]
    
        return self._all_states

    def initial_distribution(self):
        """
        Returns initial distributions over states returned be all_states
        
        Returns
        -------
        out : np.ndarray
            Initial distribution
        """

        # lazy initialization
        if self._initial_distribution is None:
            from scipy import cluster
            
            allstates = self.all_states()
            initialstate = self.initstates().__next__()

            init_index = self.get_stateindex(initialstate)

            distribution = np.zeros(len(allstates))
            distribution[init_index] = 1.0

            self._initial_distribution = distribution

        return self._initial_distribution
    

    def all_transitions_continuous(self, decstate, action):
        """
        Returns all transitions and probabilities for the given state and action.

        The returned states are continuous and are not quantized according to 
        self._all_states()

        Returns
        -------
        out : list
            Sequence of tuples: (nextstate, probability, reward)
        """

        inventory, capacity, priceindex = decstate
        priceindex = int(priceindex)

        assert(inventory >= 0 and inventory <= capacity)
        
        # determine buy and sell prices
        pricesell = self.price_sell[priceindex]
        pricebuy = self.price_buy[priceindex]
        
        # trim action based on the current inventory
        action = max(action, - inventory)
        action = min(action, capacity - inventory)
        
        # update the next inventory based on the action
        ninventory = inventory + action
        
        # compute capacity loss
        capacity_loss = self.degradation(inventory / capacity, ninventory / capacity) * capacity
        assert capacity_loss >= -1e-10, 'Cannot have negative capacity loss' 

        if self.change_capacity:
            ncapacity = max(0,capacity - capacity_loss)
            ninventory = min(ninventory, ncapacity)
        else:
            ncapacity = capacity

        # compute the reward for the transition
        reward = - (pricebuy if action >= 0 else pricesell) * action
        reward -= capacity_loss * self.capacity_cost

        # sample the next price index
        return (((ninventory,ncapacity,npriceindex),probability,reward) \
                for npriceindex, probability in \
                    enumerate(self.price_probabilities[priceindex,:]) if probability > 0)

    def all_transitions(self, stateindex, actionindex):
        """
        Returns all transitions and probabilities for the given state and action.

        The returned states are continuous and are not quantized according to 
        self._all_states()

        Parameters
        ----------
        stateindex : int
            Index of the state in the list returned by all_states
        actionindex : int
            Index of the action in the list returned by actions

        Returns
        -------
        out : sequence
            Sequence of tuples: (nextstate, probability, reward)
        """
        allstates = self.all_states()
        decstate = allstates[stateindex]

        allactions = self.actions(decstate)
        action = allactions[actionindex]

        # map transitions to the state indexes
        return [(self.get_stateindex(s),p,r) 
            for s,p,r in 
                self.all_transitions_continuous(decstate, action)]

    def transition(self,decstate,action):
        """ 
        Represents a transition from a state.

        Charging over the available capacity, or discharging below empty is not possible.
        Any action that attempts to do that is automatically limited to the capacity.

        Parameters
        ----------
        decstate : state
            inventory,capacity,priceindex
        action : float
            change in charge (this is a float value, not the index)

        Returns
        -------
        out : expectation state
            inventory,capacity,reward
        """
        #TODO: replace by a call to all_transitions
        inventory, capacity, priceindex = decstate
        assert inventory >= 0 and inventory <= capacity
        
        # determine buy and sell prices
        pricesell = self.price_sell[priceindex]
        pricebuy = self.price_buy[priceindex]
        
        # trim action based on the current inventory
        action = max(action, - inventory)
        action = min(action, capacity - inventory)
        
        # update the next inventory based on the action
        ninventory = inventory + action
        
        # compute capacity loss
        capacity_loss = self.degradation(inventory / capacity, ninventory / capacity) * capacity
        assert capacity_loss >= -1e-10, 'Cannot have negative capacity loss' 

        if self.change_capacity:
            ncapacity = max(0,capacity - capacity_loss)
            ninventory = min(ninventory, ncapacity)
        else:
            ncapacity = capacity

        # compute the reward for the transition
        reward = - (pricebuy if action >= 0 else pricesell) * action
        reward -= capacity_loss * self.capacity_cost

        # sample the next price index
        pricecount = self.price_probabilities.shape[1]
        npriceindex = np.random.choice(\
            np.arange(pricecount,dtype=int), \
            p=self.price_probabilities[priceindex,:])

        return (reward,(ninventory,ncapacity,npriceindex))

    def actions(self, state):
        """
        List of applicable actions in the state. Relative change
        in capacity
        """
        inventory, capacity, _ = state

        discharge_floor = max(-inventory,-self.discharge_limit)
        charge_ceil = min(capacity - inventory,self.charge_limit)

        return np.linspace(discharge_floor, charge_ceil, self._action_cnt)

    def initstates(self):
        """ The initial state is given by the configuration and the 1st state of the 
        price process. """
        return itertools.repeat( (self.initial_inventory,self.initial_capacity,0) )

    def price_levels(self):
        """ Returns the number of price states in the Markov model """
        return self.price_probabilities.shape[0]

class Features:
    """ 
    Suitable features for inventory management 
    """
    linear = (features.piecewise_linear(None), features.piecewise_linear(None))

## Threshold policy definitions
def threshold_policy(lowers, uppers, simulator):
    """
    Construct a threshold policy with different thresholds for different price 
    indexes.
    
    Assumes that the capacity of the battery does not change.
    
    Lower is the lower inventory target, and upper is the upper inventory target
    
    Parameters
    ----------
    lowers : list
        List of lower thresholds
    uppers : list
        List of upper thresholds
    simulator : inventory.Simulator
        Simulator of the inventory problem (used to determine available actions)
    """    
    assert len(lowers) == len(uppers)
    assert np.min(uppers - lowers) >= -1e-4
    
    def policy(state):
        inventory,capacity,priceindex = state
        
        # compute the target charge change
        if inventory < lowers[priceindex]:
            target = lowers[priceindex] - inventory # the target charge change
        elif inventory > uppers[priceindex]:
            target = uppers[priceindex] - inventory # the target charge change
        else:
            # if it is between the thresholds, then there is no change
            target = 0
        
        # find the closest (discretized) action
        actions = simulator.actions(state)
        actionindex = np.argmin(np.abs(actions - target))
        return actions[actionindex]
    
    return policy

## Threshold Optimization Functions

import math
import random
 
def _eval_dimchange(sim,lowers,uppers,dim,l,u,horizon,runs):
    """ Evaluates the dimension change impact """
    dim_lowers = lowers.copy()
    dim_uppers = uppers.copy()
    
    dim_lowers[dim] = l
    dim_uppers[dim] = u
    
    policy = raam.examples.inventory.threshold_policy(dim_lowers, dim_uppers, sim)
    
    # Common random numbers for the evaluation!
    np.random.seed(0)
    random.seed(0)
    
    samples = sim.simulate(horizon,policy,runs)
    
    print('.', end='')
    return samples.statistics(sim.discount)['mean_return']


def optimize_jointly(sim,step=0.1,horizon=600,runs=5):
    """
    Jointly optimizes uppen and lower thresholds for charging and discharging for 
    each dimension.
    
    It can be shown (a publication pending) that this method will compute
    the optimal solution when there is no degradation in the battery.
    """
    
    values = [(l,u) for l in np.arange(0,1+step/2,step) for u in np.arange(l,1+step/2,step) ]
    
    # copy the lower and upper bounds
    lowers = np.zeros(len(sim.price_buy))    # lower thresholds
    uppers = np.ones(len(sim.price_buy))     # upper thresholds
   
    for iteration in range(10):
        print('Lowers', lowers)
        print('Uppers', uppers)
        
        for dimension in range(len(sim.price_sell)):
            print('Dimension', dimension)
            returns = [_eval_dimchange(sim,lowers,uppers,dimension,l,u,horizon,runs) for (l,u) in values]
            
            maxindex = np.argmax(returns)
            
            print('\n', returns[maxindex])
            l,u = values[maxindex]
            lowers[dimension] = l
            uppers[dimension] = u
    
    print('Lowers', lowers)
    print('Uppers', uppers)


def optimize_independently(sim,step=0.1,horizon=600,runs=5):
    """
    Optimizes the upper and lower thresholds independently. It is not clear 
    that this method actually computes the optimal policy        
    """
    
    epsilon = 1e-6                  # small value to deal with numertical issues

    # copy the lower and upper bounds
    lowers = 0.5*np.ones(len(sim.price_buy))    # lower thresholds
    uppers = 0.5*np.ones(len(sim.price_buy))     # upper thresholds
    
    
    for iteration in range(10):
        print('Lowers', lowers)
        print('Uppers', uppers)
        
        weight = 1.0 / math.sqrt(iteration + 1)
        
        for dimension in range(len(sim.price_sell)):
            print('Dimension', dimension)
            
            print('   lowers')
            values = np.arange(0,1+_epsilon,step)
            if len(values) > 0:
                returns = [_eval_dimchange(sim,lowers,uppers,dimension,\
                            l,max(l,uppers[dimension]),horizon,runs)\
                             for l in values]
                maxindex = np.argmax(returns)
                l = values[maxindex]
                lowers[dimension] = weight * l + (1-weight)*lowers[dimension]
                uppers[dimension] = max(uppers[dimension],lowers[dimension])
                assert lowers[dimension] <= uppers[dimension]
            
                print('\n',returns[maxindex])
            
            print('\n   uppers')
            values = np.arange(0,1+_epsilon,step)
            if len(values) > 0:
                returns = [_eval_dimchange(sim,lowers,uppers,dimension,\
                            min(lowers[dimension],u),u,horizon,runs) \
                            for u in values]
                maxindex = np.argmax(returns)
                u = values[maxindex]
                uppers[dimension] = weight*u + (1-weight)*uppers[dimension]
                lowers[dimension] = min(lowers[dimension],uppers[dimension])
                assert lowers[dimension] <= uppers[dimension]
            
                print('\n',returns[maxindex])

    print('Lowers', lowers)
    print('Uppers', uppers)


## Plotting functions

def plot_degradation(degrad, ex_inventories = [0.1,0.5,0.9],delta=None):
    """
    Plots the degradation function for examples of the current inventory
    
    Parameters
    ----------
    degrad : fun
        Degradation function, the output of :fun:`degradation`
    ex_inventories : list, optional
        List of example inventories to use for plotting
    delta : dict
        Two delta functions (the derivative of the degradation)
    """
    
    import matplotlib.pyplot as pp
    
    x = np.linspace(0,1,100)
    
    #ax1 = pp.subplot()
    
    for ei in ex_inventories:
        y = np.array([degrad(ei, ix) for ix in x])
        pp.plot(100*x,100*y,label="$d(x,y-x)$,$x=%2.0f\\%%$" % (100*ei))
        
    #ax2 = ax1.twinx()
    
    if delta is not None:
        pp.plot(100*x, 100*delta['charge'](x), '.', label='$\\delta_+$')
        pp.plot(100*x, 100*delta['discharge'](x), '.', label='$\\delta_-$')
        
    pp.xlabel('New State of Charge (%): $y$')
    pp.ylabel('Capacity Loss (%): $d(x,y-x)$')
    
    pp.legend(loc=9)
    pp.grid()

