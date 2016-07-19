import raam
import numpy as np
from itertools import combinations
from random import random
from math import exp,sqrt

def _log(x):
    pass

config = {
    'products' : [
        {'name':'paris', 'margin':7},
        {'name':'newyork', 'margin':6},
        {'name':'bellevue', 'margin':9},
        {'name':'amherst', 'margin':15},
        {'name':'northconway', 'margin':8},
        {'name':'ashford', 'margin':18}],
    'customers' : [
        {'name':'metropolitan', 'values':[13,13.1,12.9,9,11,9],   'share':1.0}],
    'objective': 'margin',
    'recommendcount': 2
}

config2 = {
    'products' : [
        {'name':'paris', 'margin':4},
        {'name':'newyork', 'margin':1},
        {'name':'bellevue', 'margin':9},
        {'name':'amherst', 'margin':15},
        {'name':'northconway', 'margin':3},
        {'name':'ashford', 'margin':20}],
    'customers' : [
        {'name':'metropolitan', 'values':[13,13.1,12.9,9,11,9],   'share':0.6},
        {'name':'outdoorsy',    'values':[14,12,11,13,14,14], 'share':0.3},
        {'name':'mountaineer',  'values':[10,9,13,9,13,14],   'share':0.1}],
    'objective': 'value',
    'recommendcount': 2
}

def logit(values):
    p = np.exp(values)
    p = p / p.sum()
    return p

def initial_probabilities(values):
    """
    Computes the initial choice probabilities when not using the recommendations
    """
    return logit(values / 100)

def purchase_probability(value):
    """ 
    Returns the probability that the customer purchases the item of the given value 
    """
    # the value 13 is arbitrary and should be fir to data
    a = exp(13) 
    b = exp(value)
    return b / (a + b)
    
def recommendation_probabilities(currentvalue,values):
    """
    Returns the probability of taking one of the recommended actions
    
    Parameters
    ----------
    currentvalue : float
        Value of the current item
    values : array
        Values of the recommended items
    """
    a = exp(3)
    b = exp(currentvalue)
    v = np.exp(values)
    
    return v / (a + b + np.sum(v))
    
def leave_probability(value):
    """
    Probability of a customer leaving
    """
    a = exp(10)
    b = exp(value)
    return min(0.1,a/(a+b))
    
def time_spent(value):
    """

    Time spent viewing the item
    """
    return sqrt(value)
    
class Recommender(raam.Simulator):  
    """
    Represents a recommendation simulator
    
    decision state:
        - current product (index)
        - customer segment (index)
        
    expectation state:
        - current product (index)
        - customer segment (index)
        - recommended products (tuple of indices)
        
    Attributes
    ----------
    config : dict
        Configuration of the recommendation system that describes the products
        and the customers
    action_list : list
        List of actions. Each action corresponds to the list of *numbers* of the 
        products that are recommended.
    action_id : dict
        Maps actions to their ids.
    preferences : numpy.array (#segments x #products)
        The customer preferences (values)    
    objectives : numpy.array (#segments x #products)
        The value achieved when a sale is made
    prodprobs : numpy.array (#segments x #products)
        Product initial visitation probabilities based on the segment
    segmentprobs : numpy.array (#segments)
        Probability of segments
    products : list
        List of products
    customers : list
        List of customers
    enstate : decstate
        The state from which customers leave
        
    Parameters
    ----------
    config : dict
        Configuration of the recommendation system that describes the products
        and the customers

    """
    def __init__(self, config):
        self.config = config
        
        products = config['products']
        self.products = products
        customers = config['customers']
        self.customers = customers
        
        self.action_list = np.array([()] + [tuple(x) for x in combinations(range(len(products)),config['recommendcount'])])
        self.action_id = {a:i for i,a in enumerate(self.action_list)}
        
        pcount = len(products)
        
        self.preferences = np.array([[s['values'][p] for p in range(pcount)] for s in customers])
        if config['objective'] == 'value':
            self.objectives = np.array([[s['values'][p] for p in range(pcount)] for s in customers])
        elif config['objective'] == 'margin':
            self.objectives = np.array([[p['margin'] for p in products] for s in customers])
        else:
            raise ValueError('Unknown objective type.')
        # product probailities
        self.prodprobs = np.array([initial_probabilities(self.preferences[s,:]) for s in range(len(customers))])
        # segment probabilities
        self.segmentprobs = np.array([c['share'] for c in customers])
        if abs(self.segmentprobs.sum() - 1) > 0.01:
            raise ValueError('Segment probabilities must sum to 1')
    
        self.endstate = (pcount,len(customers))
    
    @property
    def discount(self):
        return 1.0
        
    def transition(self,decstate,action):
        """ Computes a transition """
        product, segment = decstate 
        
        currvalue = self.preferences[segment,product]
        _log('considering product %d' % product)
        
        # decide whether to buy
        purch_prob = purchase_probability(currvalue)
        if random() <= purch_prob:
            _log('** purchasing %d' % product)
            return self.objectives[segment,product],self.endstate
        # decide whether to take a recommendation
        if len(action) > 0:
            _log('considering recommendation %s' % str(action))
            recomvalues = [self.preferences[segment,a] for a in action]
            recom_prob = recommendation_probabilities(currvalue, recomvalues)
            recom_prob_sum = np.sum(recom_prob)
            if random() <= recom_prob_sum:
                _log('taking recommendation %s' % str(action))
                chosen = action[np.random.choice(len(recom_prob), p=(recom_prob / recom_prob_sum))]
                return 0,(chosen,segment)
        # decide whether to leave
        leave_prob = leave_probability(currvalue)
        if random() <= leave_prob:
            _log('** leaving')
            return 0, self.endstate
        # decide whether to just look at a different product
        _log('resetting product choice')
        return 0,(np.random.choice(self.prodprobs.shape[1], p=self.prodprobs[segment,:]),segment)
        
    def end_condition(self,decstate):
        product, segment = decstate
        return product >= len(self.products) or segment >= len(self.customers)
        
    def initstates(self):
        while True:
            customer = np.random.choice(len(self.segmentprobs), p=self.segmentprobs)
            product = np.random.choice(self.prodprobs.shape[1], p=self.prodprobs[customer,:])
            assert customer < len(self.segmentprobs)
            yield (product,customer)
        
    def actions(self,decstate):
        return self.action_list

def addsales(stats):
    stats['sales'] = sum(1 if r > 0.01 else 0 for r in stats['sum_rewards'])

def unoptimized(simulator,horizon,runs,type):
    """
    Runs simulations using unoptimized policies
    
    Parameters
    ----------
    simulator : raam.examples.recommender.Recommender
    horizon : int
    runs : int 
        Number of runs to simulate (results are averaged)
    type : {"none", "random", "myopic", "liked"}
        Type of policy to run
    Return
    ------
    samples : Samples
        Raw samples
    stats : dict
        Processed statistics
    """    
    if type == 'none':
        policy = lambda decstate: simulator.actions(decstate)[0]
    elif type == 'random':
        policy = simulator.random_policy()
    elif type == 'myopic':
        policy = simulator.greedy_policy_q(lambda x: 0)
    elif type == 'liked':
        def policy(decstate):
            product, segment = decstate
            best_product = np.argmax(simulator.preferences[segment,:])
            return simulator.action_list[best_product + 1]
    else:
        raise ValueError('Invalid type') 
            
    samples = simulator.simulate(horizon,policy,runs)
    stats = samples.statistics(simulator.discount)
    addsales(stats)
    return samples,stats


def optimized(simulator,samples,testhorizon,testruns,type,solver='c',norm='none',iterations=1000,residual=0.0001,robmatall=None):
    """
    Computes an optimized policy and evaluates its performance
    
    Parameters
    ----------
    simulator : raam.examples.recommender.Recommender
    samples : raam.Samples
        Samples used to construct the MDP
        Ignored when robmat is provided
    testhorizon : int
    testruns : int
    type : {'known', 'p2p'}
        The type optimization
        Ignored when robmat is provided
    solver : {'c', 'py'}
        The type of the solver to be used, either the python or c implementation
    norm : {'none', ('l1',threshold)}
        What norm to use for the solution and potentially the threshold
    iterations : int
        Number of iterations 
    robmatall : (function,function,dict)
        An optional cache output that preserves the source of the samples
        (decagg,expagg,matrices)
        
    Return
    ------
    samples : Samples
        Raw samples
    stats : dict
        Processed statistics
    values : np.ndarray
        Value function
    policy_vec : np.ndarray
        Policy with action indices
    robmatall : (function,function,dict)
        An optional cache output that preserves the source of the samples
        (decagg,expagg,matrices)
    """
    from raam import robust
    from raam import crobust

    iterations = 1000

    if robmatall is None:    
        if type == 'known':
            decagg = robust.IndexCache()
            expagg = robust.IdCache()
        elif type == 'p2p':
            decagg = robust.__ind__
            expagg = robust.IdCache()
        else:
            raise ValueError('Invalid type.')        
            
        robmat = robust.matrices(samples,decagg=decagg,expagg=expagg)
    else:
        decagg,expagg,robmat = robmatall
    
    if solver == 'c':
        statecount = robmat['dectoexp'].shape[0]
        rmdp = crobust.RoMDP(statecount)
        rmdp.from_sample_matrices(robmat['dectoexp'], robmat['exptodec'], robmat['actions'], robmat['rewards'])
        #print(str(rmdp.to_string()).replace('\\n','\n'))
        
        if norm == 'none':
            values, policy_vec, residual = rmdp.value_iteration_replace(np.zeros(statecount), \
                                    simulator.discount, iterations, stype=0, maxresidual=residual)
        elif len(norm) == 2 and norm[0] == 'l1':
            rmdp.set_thresholds(norm[1])
            values, policy_vec, residual = rmdp.value_iteration_replace_l1(np.zeros(statecount), \
                                    simulator.discount, iterations, stype=0, maxresidual=residual)
        else:
            raise ValueError('Invalid norm')
        
        policy_optimized = robust.decvec2policy(policy_vec, robmat['actions'], lambda ds: decagg(ds,add=False)[0])
        
    elif solver == 'py':
        if norm == 'none':
            values, policy_vec, residual = robust.valiter(robmat['dectoexp'],\
                                    robmat['exptodec'],robmat['rewards'],\
                                    simulator.discount,iterations)                                
        elif len(norm) == 2 and norm[0] == 'l1':
            values, policy_vec, residual = robust.valiter(robmat['dectoexp'],\
                                robmat['exptodec'],robmat['rewards'],simulator.discount,iterations,\
                                uncertainty='l1',distributions=distributions,bounds=bounds)                                            
        else:
            raise ValueError('Invalid norm')
        
        policy_optimized = robust.vec2policy(policy_vec, robmat['actions'], lambda ds: decagg(ds,add=False)[0])
            
    else:
        raise ValueError('Invalid solver')
        

                            
    samples_optimized = raam.simulate(simulator,testhorizon,policy_optimized,testruns)
    stats_optimized = samples_optimized.statistics(simulator.discount)
    addsales(stats_optimized)
    return samples_optimized,stats_optimized,values,policy_vec,(decagg,expagg,robmat)

def config_from_matrix(W,init,products,segments,margins,objective='margin',recommendcount=1):
    """
    Costruct the configuration file from a factorized matrix
    
    m - number of segments
    n - number of products
    
    Parameters
    ----------    
    W : matrix m x n
        Segment to product preference matrix
    init : list (n)
        Distribution of segments
    products : list (n)
        List of product names
    segments : list (m)
        List of segment names
    margins : list (n)
        The margin values for each product
    objective : {'value', 'margin'}
        The objective, either the customer value of the product margin
    recommendcount : int
        Number of recommendations to make

    Returns
    -------
    out : dict
        Configuration file
    """
    (m,n) = W.shape
    assert len(init) == m and len(products) == n and len(segments) == m and len(margins) == n
    assert np.isclose(np.sum(init), 1) and np.min(init) 
    
    products = [{'name' : p, 'margin' : m} for p,m in zip(products,margins)]
    customers = [{'name' : s, 'values':v, 'share':i} for s,v,i in zip(segments,W,init)]
    
    return {'products':products, 'customers':customers,'objective':objective,'recommendcount':recommendcount}
        
