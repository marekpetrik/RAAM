import numpy as np
import scipy as sp
import itertools
import raam
import pickle
from raam import features

epsilon = 1e-10

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
        This is not an index but an absolute value of the change in charge

    Parameters
    ----------
    config : dict
        Configuration. See DefaultConfiguration for an example
    discount : float, optional
        Discount factor
    action_step : float, optional
        Discretization step for actions. This is an absolute number and should
        scale with the size of the storage.
    """

    def __init__(self,config,discount=0.9999,action_step=0.05):
        """ 
        If no filename and no configuration are provided, 
        then the configuration is created automatically 
        """
        
        self._discount = discount
        self._action_step = action_step
        
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
        return self._discount

    def transition_dec(self,decstate,action):
        """ 
        Represents a transition from the deterministic state to an expectation state.

        Parameters
        ----------
        decstate : decision state
            inventory,capacity,priceindex
        action : float
            change in charge (this is a float value, not the index)

        Returns
        -------
        out : expectation state
            inventory,capacity,reward
        """
        inventory, capacity, priceindex = decstate
        
        pricesell = self.price_sell[priceindex]
        pricebuy = self.price_buy[priceindex]
        
        assert inventory >= 0
        assert inventory <= capacity
        
        action = max(action, - inventory)
        action = min(action, capacity - inventory)
        
        ninventory = inventory + action
        capacity_loss = self.degradation(inventory / capacity, ninventory / capacity) * capacity
        assert capacity_loss >= -1e-10
        if self.change_capacity:
            ncapacity = max(0,capacity - capacity_loss)
            ninventory = min(ninventory, ncapacity)
        else:
            ncapacity = capacity

        reward = - (pricebuy if action >= 0 else pricesell) * action
        reward -= capacity_loss * self.capacity_cost
        return (ninventory,ncapacity,priceindex,reward)

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
        inventory,capacity,priceindex,reward = expstate
        
        assert inventory >= 0
        assert inventory <= capacity

        pricecount = len(self.price_probabilities[priceindex,:])

        priceindex = np.random.choice(\
            np.arange(pricecount,dtype=int), \
            p=self.price_probabilities[priceindex,:])
        
        return (reward,(inventory,capacity,priceindex))

    def actions(self,decstate):
        """
        List of applicable actions in the state
        """
        inventory, capacity, _ = decstate
        return np.arange(-inventory, capacity - inventory + epsilon, self._action_step)

    def all_actions(self):
        """
        List of all possible actions (charge and discharge capacities)
        """
        return np.arange(-self.initial_capacity, self.initial_capacity + epsilon, \
                            self._action_step)

    def stateiterator(self):
        expstate = (self.initial_inventory,self.initial_capacity,0,0)
        while True:
            yield self.transition_exp(expstate)[1]

    def initstates(self):
        return self.stateiterator()

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
 
_epsilon = 1e-6

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


## End to end construction helper functions

def makesimulator(discretization,discount,action_step=0.1):
    """
    Constructs an inventory simulator and helper functions to simplify the 
    construction of the sampled MDP. The prices are constructed using the 
    martingale price evolution.
    
    The inventory level is discretized according to the provided parameter, but 
    the discretization happens after simulation just for the samples.

    Parameters
    ----------
    discretization : int
        Number of discrete inventory levels 
    
    Returns
    -------
    sim : raam.Simulator
        Simulation object
    create_srmdp : function
        Function that creates an srmdp from the samples
    create_policy : function
        Function that creates a function policy from the result vector
    """
    
    from raam import crobust
    
    epsilon = 1e-6                  # small value to deal with numertical issues
    
    # problem configuration
    config = raam.examples.inventory.configuration.construct_martingale(np.arange(5), 5)
    config['change_capacity'] = True        # updates capacity of inventory (degradation)
    sim = raam.examples.inventory.Simulator(config,action_step=action_step,discount=discount)
    
    # construct the set of possible actions and map them to indexes
    all_actions = sim.all_actions()
    
    def decmap(s):
        soc,capacity,priceindex = s
        assert type(priceindex) == int or type(priceindex) == np.int64
        return (soc,priceindex)
        
    def expmap(s):
        soc,capacity,priceindex,reward = s
        return (soc,capacity,reward)
    
    #** for narrow samples
    # decision state: soc, priceindex
    decagg_big = raam.features.GridAggregation( \
                        ((0,config['initial_capacity']+epsilon), (0,config['price_sell'].shape[0])),\
                        (discretization, config['price_sell'].shape[0]) )
    
    # expectation state: soc, priceindex, reward 
    expagg = raam.features.GridAggregation( \
                        ((0,config['initial_capacity']+epsilon), (0,config['price_sell'].shape[0]), (-5,5)), \
                        (3*discretization, config['price_sell'].shape[0], 200) )
    
    # assign an index to a decision states
    dab = lambda x: decagg_big.classify(x,True)     
    # assign an index to an expectation state
    ea = lambda x: expagg.classify(x,True)
    # represents the individual states (e.g. for computing worst case or best case solutions)
    das = lambda x: 0
    # represents individual expectation states
    eas = lambda x: 0
    # aggregations of actions
    aa = lambda x : np.argmin(np.abs(all_actions - x))

    # a helper function that constructs SRMDP from the provided samples
    def create_srmdp(samples):
        narrow_samples = raam.samples.SampleView(samples,decmap=decmap,
                    expmap=expmap, actmap=raam.identity)
        srmdp = crobust.SRoMDP(0,sim.discount)
        srmdp.from_samples(narrow_samples,dab,das,ea,eas,aa)
        return srmdp
        
    def create_policy(srmdp, policy_vec):
        policy_vec_dec = srmdp.decpolicy(len(decagg_big), policy_vec)
        policy = raam.vec2policy(policy_vec_dec, all_actions, lambda x: dab(decmap(x)),0)
        return policy
        
    return sim, create_srmdp, create_policy
