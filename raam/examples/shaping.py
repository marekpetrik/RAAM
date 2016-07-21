"""
=================================================
Demand shaping (:mod:`raam.examples.shaping`)
=================================================

Models an inventory with actions that enable shifting demand 
among different products.

"""
import numpy as np
import scipy as sp
import itertools
import raam
import pickle
import math
from raam import features


def policyNoAction(decstate):
    """ No shaping action is taken """
    return set([])

class ConfigSimple:
    """
    Simple configuration of the demand shaping problem
    with a randomly generated product valuations

    """
    def __init__(self, products=3):
        self.product_types = ['product_' + str(p) for p in range(products)]
        self.customer_types = ['customer_' + str(c) for c in range(products)]
        self.customer_probabilities = sp.random.dirichlet(np.ones(len(self.customer_types)))
        # each action type is a subset of products that are shaped
        self.action_types = ['promote_' + str(p) for p in self.product_types]
        # the values that the customers assign to products assuming
        # a multinomial logit model of their behavior
        # base values for all products
        self.base_valuations = products*np.diag(np.random.rand(len(self.customer_types))) 
        # profits
        self.profits = np.random.rand(len(self.product_types))
        # first dimension: action, second dimension: product influenced
        self.action_valuation_delta = np.diag(np.random.rand(len(self.product_types)))
        # action price - only applied when the product is purchased
        self.action_price = np.random.rand(len(self.action_types)) * 0
        # replenishment probabilities
        self.replenishement_probabilities = sp.random.dirichlet(np.ones(len(self.product_types)))
        self.initial_inventory = np.ones(len(self.product_types)) * products 
        self.initial_inventory.flags.writeable = False
        self.initial_customer = 0    
        self.decstate_init = self.initial_customer,self.initial_inventory
        assert len(self.product_types) == len(self.customer_types)
        # the valuation delta of each action
        assert len(self.action_types) == len(self.product_types) # TEMPORARY: a single action per product right now


    def purchase_probability(self,customer,action):
        """ 
        Take the customer type and the action that recommends products

        Parameters
        ----------
        customer : int
            Customer type
        action : set(int)
            The set of the number of products that are recommended
                    the number of of the action

        Returns
        -------
        out : list
            Purchase probabilities for all individual products
        """
        assert customer >= 0 and customer < len(self.customer_types)
        assert len(action) == 0 or (min(action) >= 0 and max(action) < len(self.action_types)), 'Invalid action: "%s"' % action

        value = self.base_valuations[customer,].copy()
        for a in action:
            value += self.action_valuation_delta[a,]
        value_exp = np.exp(value)
        return value_exp / np.sum(value_exp)

class Simulator(raam.Simulator):
    """
    Simulates demand shaping and customer choice

    There are two types of states. Expectation and decision states.
    -  Decision state: A tuple representation with (customer, inventory)
            * customer: the type of the customer (int) 
            * inventory: current inventory (vector)
    - Expectation state: The expectation state with (purchase probabilities, reward, inventory)
            * purchase probabilities - cumulative sum (vector)
            * reward for the action (float)

    The simulator works with a wrapper state (a tuple):
        (is_decision:bool, inner state)
    """

    def __init__(self, filename=None,config=None):
        """ If no filename and no configuration are provided, 
        then the configuration is created automatically """
        if filename is None:
            self.config = config if config is not None else ConfigSimple()
        else:
            self.load_config(filename)

    def load_config(self,filename):
        """ Loads configuration """
        with open(filename, mode='rb') as fileinput:
            self.config = pickle.load(fileinput)
        
    def save_config(self,filename):
        """ Saves configuration """
        with open(filename, mode='wb') as fileoutput:
            pickle.dump(self.config,fileoutput)

    @property
    def discount(self):
        # the sqrt is to compensate for the virtual step due to 
        # separate decision and expectation states
        return math.sqrt(0.95)

    def transition(self, state, action):
        """
        Transition from either expectation or decision state.
        """
        is_decision, inner_state = state

        if is_decision:
            # the following state must be an expectation one
            newstate = (False, self.transition_dec(inner_state, action))
            reward = 0
        else:
            reward, newinnerstate = self.transition_exp(inner_state)
            newstate = (True, newinnerstate)
        
        return reward, newstate


    def transition_dec(self,decstate,action):
        """ 
        Represents a transition from the deterministic state to an expectation state.
        Decision state - customer type and inventory
        Expectation state - inventory 

        Parameters
        ----------
        decstate : decision state
            A tuple representation with (customer, inventory)
        action : set(int) 
            The set of the number of products that are recommended
                    the number of of the action
        Returns
        -------
        out : expectation state
            The expectation state with (purchase probabilities, reward, inventory)
        """
        customer, inventory = decstate
        assert len(inventory) == len(self.config.product_types)
        assert customer >= 0 and customer < len(self.config.customer_types)

        purchprobs = self.config.purchase_probability(customer, action)
        profitaction = 0
        for a in action:
            profitaction -= self.config.action_price[a]
        return (purchprobs, profitaction, inventory)

    def transition_exp(self,expstate):
        """ 
        Transition from an expectation state to a decision state

        Parameters
        ----------
        expstate : expectation state

        Returns
        -------
        out : tuple(reward, decision state)
            * Reward is a float reward
            * Decision states is a tuple(customer, inventory)
        """
        purchprobs, profitaction, inventory = expstate
        inventory = inventory.copy()
        assert len(inventory) == len(self.config.product_types)
        assert len(purchprobs) == len(inventory)
        assert abs(np.sum(self.config.customer_probabilities) - 1) < 0.0001, \
            "Probabilities should sum to 1 instead of %f " % np.sum(self.config.customer_probabilities)
        assert abs(np.sum(purchprobs) - 1) < 0.0001, \
            "Probabilities should sum to 1 instead of %f " % np.sum(purchprobs)

        customer = np.random.choice(np.array(range(len(self.config.customer_probabilities))),\
                        p=self.config.customer_probabilities)
        product = np.random.choice(np.array(range(len(purchprobs))), p=purchprobs)
        
        if inventory[product] >= 1:
            profit = self.config.profits[product] + profitaction
            inventory[product] -= 1
        else:
            profit = 0.0
        refresh = np.random.choice(np.array(range(len(self.config.replenishement_probabilities))),\
                p=self.config.replenishement_probabilities)
        inventory[refresh] += 1
        return (profit, (customer, inventory))

    def actions(self,state):
        """
        Available actions in decision states are configured. And there is only 
        a single action available in an expectation state.
        """
        is_decision, _ = state
        if is_decision:
            return [frozenset()] + [frozenset([i]) for i,t in enumerate(self.config.action_types)] 
        else:
            return [frozenset()]

    def initstates(self):
        """ There is a single initial state """
        c = self.config
        return itertools.repeat( (True,(c.initial_customer, c.initial_inventory)) )

def decstate_rep(decstate):
    """ Computes the representation of a decision state """
    customer, inventory = decstate
    return [customer] + inventory.tolist() 

def expstate_rep(expstate):
    """ Computes the representation of an expectation state """
    purchprobs, profitaction, inventory = expstate
    return list(purchprobs) + [profitaction] + list(inventory)

Representation = {'decmap' : decstate_rep, 'expmap' : expstate_rep}

class Features:
    linear = (features.piecewise_linear(None), features.piecewise_linear(None))
