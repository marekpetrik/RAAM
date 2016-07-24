"""
=================================
Unit tests (:mod:`raam.test`)
=================================
"""
import raam
from raam import examples
import unittest
import random
import numpy as np
import itertools
import json
from operator import itemgetter
import math

import craam

from raam import features
from raam import robust

settings = {}

try:
    from optimadp import opl
    settings['opl'] = True
except:
    settings['opl'] = False

try:
    from optimadp import direct
    settings['cvxopt'] = True
except:
    settings['cvxopt'] = False

def create_test_sample():
    """ 
    Creates a test in-memory sample and returns it 
    
    See Also
    --------
    raam.samples.MemSamples
    """
    
    samples = raam.MemSamples()
    
    s_decstates = [[0,1],[1,2],[2,3]]
    profits = [1,2,3]
    
    for i,(d,p) in enumerate(zip(s_decstates, profits)):
        samples.add_sample(raam.Sample(d, 1, d, p, 1.0, i, 0))
    return samples

class BasicTests(unittest.TestCase):
    """ Tests basics """
    def test_representation(self):
        q = create_test_sample()
        s = q.validate()
        r = q.statistics(0.95)['mean_return']
        self.assertAlmostEqual(5.6075, r)
        self.assertEqual(3, s['samples'])

class BasicTestsSimulation(unittest.TestCase):
    """ Simulation tests """

    def setUp(self):
        self.horizon = 30

    def test_check_simulation(self):
        random.seed(1000)
        np.random.seed(1000)
        sim = raam.examples.shaping.Simulator()
        result = sim.simulate(2,raam.examples.shaping.policyNoAction,1)
        ret = result.statistics(sim.discount)['mean_return']
        stats = result.validate()
        self.assertEqual(2, stats['samples'])
        self.assertAlmostEqual(0.039678834047865277, ret)

    def test_check_simulation_multiple(self):
        random.seed(1000)
        np.random.seed(1000)
        sim = raam.examples.shaping.Simulator()
        result = sim.simulate(2*self.horizon,raam.examples.shaping.policyNoAction,range(5))
        ret = result.statistics(sim.discount)['mean_return']
        stats = result.validate()
        self.assertAlmostEqual(2.9236894010933661, ret, 4)
        self.assertEqual(2*self.horizon*5, stats['samples'])
        
    def test_check_simulation_multiple_counter(self):
        random.seed(1000)
        np.random.seed(1000)
        horizon = 30
        sim = examples.counter.Counter()
        result = sim.simulate(horizon,sim.random_policy(),5)
        ret = result.statistics(sim.discount)['mean_return']
        self.assertAlmostEqual(-9.480222759759355, ret, 4)
        stats = result.validate()
        self.assertEqual(30*5, stats['samples'])

    def test_check_simulation_merge(self):
        sim = raam.examples.shaping.Simulator()
        result1 = sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,1)
        result2 = sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,[2])
        stats1 = result1.validate()
        self.assertEqual(1, stats1['runs'])
        self.assertEqual(30, stats1['samples'])
        stats2 = result2.validate()
        self.assertEqual(1, stats2['runs'])
        self.assertEqual(30, stats2['samples'])
        result = raam.samples.MemSamples()
        result.merge(result1)
        result.merge(result2)
        stats = result.validate()
        self.assertEqual(2, stats['runs'])
        self.assertEqual(60, stats['samples'])

    def test_generate_exp_samples_ofexp(self):
        sim = raam.examples.shaping.Simulator()
        result = sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,1)
        extended = sim.sample_transitions(result,10)
        stats = extended.validate()
        self.assertEqual(330, stats['samples'])

    def test_generate_exp_samples_ofdec(self):
        sim = raam.examples.shaping.Simulator()
        result = sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,1)
        extended = sim.sample_transitions(result,10)
        stats = extended.validate()
        self.assertEqual(330, stats['samples'])

    def test_generate_dec_samples_ofexp(self):
        sim2 = examples.pendulum.Simulator2()
        result = sim2.simulate(30,sim2.random_policy(),1)
        extended = sim2.sample_actions_transitions(result)
        stats = extended.validate()
        # this one does not check for duplicates
        self.assertEqual(120, stats['samples'])

    def test_generate_dec_samples_ofdec(self):
        sim2 = examples.pendulum.Simulator2()
        result = sim2.simulate(30,sim2.random_policy(),1)
        extended = sim2.sample_actions_transitions(result)
        stats = extended.validate()
        self.assertEqual(120, stats['samples'])

    def test_generate_exp_samples_ofexp_replace(self):
        sim = raam.examples.shaping.Simulator()
        result = sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,1)
        extended = sim.sample_transitions(result,10,append=False)
        stats = extended.validate()
        self.assertEqual(300, stats['samples'])

    def test_generate_exp_samples_ofdec_replace(self):
        sim = raam.examples.shaping.Simulator()
        result = sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,1)
        extended = sim.sample_transitions(result,10,append=False)
        stats = extended.validate()
        self.assertEqual(300, stats['samples'])

    def test_generate_dec_samples_ofexp_replace(self):
        sim2 = examples.pendulum.Simulator2()
        result = sim2.simulate(30,sim2.random_policy(),1)
        extended = sim2.sample_actions_transitions(result,append=False)
        stats = extended.validate()
        self.assertEqual(90, stats['samples'])

    def test_generate_dec_samples_ofdec_replace(self):
        sim2 = examples.pendulum.Simulator2()
        result = sim2.simulate(30,sim2.random_policy(),1)
        extended = sim2.sample_actions_transitions(result,append=False)
        stats = extended.validate()
        self.assertEqual(90, stats['samples'])

    def test_greedy_policy(self):
        sim = raam.examples.shaping.Simulator()
        actions = list(itertools.permutations(range(len(sim.config.action_types)),1)) + [set()]
        greedy = sim.greedy_policy_v(lambda x:0, 10)
        result = sim.simulate(self.horizon,greedy,1)
        stats = result.validate()

        self.assertEqual(30, stats['samples'])

    def test_greedy_better(self):
        sim = raam.examples.shaping.Simulator()
        actions = list(itertools.permutations(range(len(sim.config.action_types)),1)) + [set()]
        greedy = sim.greedy_policy_v(lambda x:0, 10)
        runs = range(10)

        result_greedy = sim.simulate(self.horizon,greedy,runs)
        result_noaction = sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,runs)

        stats_greedy = result_greedy.validate()
        self.assertEqual(300, stats_greedy['samples'])
        stats_noaction = result_noaction.validate()
        self.assertEqual(300, stats_noaction['samples'])
        
from collections import Counter
class SimulationTerminationTests(unittest.TestCase):
    
    def setUp(self):
        self.runs = 2000
        self.probterm = 0.1

    @staticmethod
    def correct_probability(state, discount):
        """ The correct visitation probability of the state """
        return discount**state 

    def test_stateless(self):
        random.seed(1982)
        c = raam.examples.counter.Counter(succ_prob=1)
        samples = c.simulate(1000,  lambda s: c.actions(s)[0], self.runs, probterm=self.probterm)
        decsamples = samples.samples()
        decstates = [decsample.statefrom for decsample in decsamples]
        
        counter = Counter(decstates)
        for state, frequency in counter.items():
            self.assertTrue(abs(SimulationTerminationTests.correct_probability(state, 1-self.probterm) \
                - frequency / self.runs) < 0.02)
    
class SimulationSamplesTerminationTests(unittest.TestCase):

    def setUp(self):
        self.runs = 2000
        self.steps = 2000

    def test_stateless_small(self):
        c = raam.examples.counter.Counter(succ_prob=1)
        samples = c.simulate(self.steps,  lambda s: c.actions(s)[0], self.runs,\
                    transitionlimit=50)
        self.assertEqual(50,len(tuple(samples.samples()))) 

    def test_stateless(self):
        c = raam.examples.counter.Counter(succ_prob=1)
        samples = c.simulate(self.steps,  lambda s: c.actions(s)[0], self.runs,\
                    transitionlimit=200)
        self.assertEqual(200,len(tuple(samples.samples()))) 


class TestPrecise(unittest.TestCase):
    """ Test precise MDP solvers """
    def setUp(self):
        self.samples = examples.chain.simple_samples(7,1)
        self.samplesd = features.DiscreteSampleView(self.samples)
        self.samplessmall = examples.chain.simple_samples(3,1)
        self.samplessmalld = features.DiscreteSampleView(self.samplessmall)
        self.samplesstoch = examples.chain.simple_samples(7,0.75)
        self.samplesstochd = features.DiscreteSampleView(self.samplesstoch)

    
    def test_crobust_stoch(self):
        m = craam.MDP(7, 0.9)
        for s in self.samplesstochd.samples():
            m.add_transition(s.statefrom, s.action, s.stateto, s.weight, s.reward)
        valuefunction,_,_ ,_= m.vi_gs(200)
    
        des = [26.0562239092526, 25.41454564669827, 23.67375336867879, 22.30637803181092, 23.67375336867879, 25.41454564669827, 26.0562239025266]
        for a,b in zip(des, valuefunction):
            self.assertAlmostEqual(a,b,3)     
    
    def test_crobust_stoch_re(self):
        rm = craam.RMDP(7, 0.9)
        
        # add fake transitions for all outcomes (weight ~ 0, and all outcomes )
        states = {sam.statefrom for sam in self.samplesstochd.samples()} | {sam.stateto for sam in self.samplesstochd.samples()}
        actions = {sam.action for sam in self.samplesstochd.samples()}
        
        # add transitions from all outcomes to all states, just to make sure that there
        # is one for each outcome
        for sfrom in states:
            for a in actions:
                for sto in states:
                    rm.add_transition(sfrom,a,sto,sto,0.00000001,0)
        
        statecount = len(states)
        
        distributions = [np.zeros((statecount,statecount)) for a in actions]
        
        for s in self.samplesstochd.samples():
            # the transition from the outcome to the action is 1
            rm.add_transition(s.statefrom, s.action, s.stateto, s.stateto, 1, s.reward)
            # set the distribution
            distributions[s.action][s.statefrom, s.stateto] = s.weight
        
        # set the distributions to outcomes
        for s in states:
            for a in actions:
                rm.set_distribution(s,a,distributions[a][s,:])
        
        #import pprint
        #import json
        #pprint.pprint(json.loads(rm.to_json()))
        
        
        rm.set_uniform_thresholds(0)
        valuefunction,_,_,_,_  = rm.vi_gs(200)
        des = [26.0562239092526, 25.41454564669827, 23.67375336867879, 22.30637803181092, 23.67375336867879, 25.41454564669827, 26.0562239025266]
        for a,b in zip(des, valuefunction):
            self.assertAlmostEqual(a,b,3)     
                
        rm.set_uniform_thresholds(2)
        valuefunction,_,_,_,_  = rm.vi_gs(200,stype=1)
        des = [ 30., 30., 29., 27.1, 29., 30., 30.]    
        for a,b in zip(des, valuefunction):
            self.assertAlmostEqual(a,b,3)     
          

class RobustTests(unittest.TestCase):
    """ Test robust methods"""
    
    def setUp(self):
        pass
        
    def test_idcache(self):
        states = [[0,1],[1,1],[1,2],[1,3],[1,1],[1,1]]
        results = [0,1,2,3,1,1]
        c = raam.features.IdCache()
    
        # test addition
        for s,r in zip(states,results):
            self.assertEqual(c(s),r)
    
        # test retrieval
        for s,r in zip(states,results):
            self.assertEqual(c(s),r)
        
    def test_indexcache(self):
        states = [[0,1],[1,1],[1,2],[1,3],[1,1],[1,1]]
        results = [(0,0),(1,0),(2,0),(3,0),(1,0),(1,0)]
        c = raam.features.IndexCache()
    
        # test addition
        for s,r in zip(states,results):
            self.assertEqual(c(s),r)
    
        # test retrieval
        for s,r in zip(states,results):
            self.assertEqual(c(s),r)
            
    def test_indexcache_two(self):
        states = [[0,1],[1,1],[2,2],[3,3],[1,2],[1,3]]
        results = [(0,0),(1,0),(2,0),(3,0),(1,1),(1,2)]
        aggregation = lambda x: x[0]
        c = features.IndexCache(aggregation=aggregation)
    
        # test addition
        for s,r in zip(states,results):
            self.assertEqual(c(s),r)
    
        # test retrieval
        for s,r in zip(states,results):
            self.assertEqual(c(s),r)
            
    def test_l1_worstcase(self):
        q = np.array([0.4, 0.3, 0.1, 0.2])
        z = np.array([1,2,5,4])
        t = 0
        self.assertEquals(robust.worstcase_l1(z,q,t),q.dot(z))
        
        q = np.array([0.4, 0.3, 0.1, 0.2])
        z = np.array([1,2,5,4])
        t = 1
        self.assertAlmostEqual(robust.worstcase_l1(z,q,t),1.1)        
        
        q = np.array([0.4, 0.3, 0.1, 0.2])
        z = np.array([1,2,5,4])
        t = 2
        self.assertEquals(robust.worstcase_l1(z,q,t),np.min(z))
        
        q = np.array([1.0,0.0])
        z = np.array([2,1])
        t = 0
        self.assertEquals(robust.worstcase_l1(z,q,t),2)
        
        q = np.array([1.0,0.0])
        z = np.array([2,1])
        t = 1
        self.assertEquals(robust.worstcase_l1(z,q,t),1.5)
        
        q = np.array([1.0,0.0])
        z = np.array([2,1])
        t = 2
        self.assertEquals(robust.worstcase_l1(z,q,t),1)
        
    def test_l1_worstcase_cpp(self):
        q = np.array([0.4, 0.3, 0.1, 0.2])
        z = np.array([1.0,2.0,5.0,4.0])
        t = 0
        self.assertEquals(craam.cworstcase_l1(z,q,t),q.dot(z))
        
        q = np.array([0.4, 0.3, 0.1, 0.2])
        z = np.array([1.0,2.0,5.0,4.0])
        t = 1
        self.assertAlmostEqual(craam.cworstcase_l1(z,q,t),1.1)        
        
        q = np.array([0.4, 0.3, 0.1, 0.2])
        z = np.array([1.0,2.0,5.0,4.0])
        t = 2
        self.assertEquals(craam.cworstcase_l1(z,q,t),np.min(z))
        
        q = np.array([1.0,0.0])
        z = np.array([2.0,1.0])
        t = 0
        self.assertEquals(craam.cworstcase_l1(z,q,t),2)
        
        q = np.array([1.0,0.0])
        z = np.array([2.0,1.0])
        t = 1
        self.assertEquals(craam.cworstcase_l1(z,q,t),1.5)
        
        q = np.array([1.0,0.0])
        z = np.array([2.0,1.0])
        t = 2
        self.assertEquals(craam.cworstcase_l1(z,q,t),1)
    
    def test_l1_worstcase_masked(self):
        z = np.ma.array([np.ma.masked,2,np.ma.masked,1,np.ma.masked])
        q = np.array([1,1,1,0,1]) / 4
        
        t = 0
        self.assertEquals(robust.worstcase_l1(z,q,t),2)
        
        t = 1
        self.assertEquals(robust.worstcase_l1(z,q,t),1.5)
        
        t = 2
        self.assertEquals(robust.worstcase_l1(z,q,t),1)
        
    def test_precise_solution(self):
        
        states = 100
        np.random.seed(1000)
        
        P1 = np.random.rand(states,states)
        P1 = np.diag(1/np.sum(P1,1)).dot(P1)
        P2 = np.random.rand(states,states)
        P2 = np.diag(1/np.sum(P2,1)).dot(P2)
        r1 = np.random.rand(states)
        r2 = np.random.rand(states)
        
        transitions = np.dstack((P1,P2))
        rewards = np.column_stack((r1,r2))
        actions = np.array((0,1))
        outcomes = np.array((0,0))
        
        rmdp = craam.MDP(states,0.99)
        rmdp.from_matrices(transitions,rewards)
        value,policy,residual,iterations = rmdp.mpi_jac(1000)
         
        target_value = [ 67.48585933,  67.6855307 ,  67.15995444,  67.33964064,
            67.35730334,  67.448749  ,  67.38176967,  67.65606086,
            67.4213027 ,  67.47155931,  67.07684382,  67.74003882,
            67.47210723,  67.55348594,  67.79191643,  67.2560147 ,
            67.32406201,  67.07662814,  67.203813  ,  67.33651601,
            67.39629876,  67.54799239,  67.75385862,  67.44879403,
            67.60544142,  67.40952776,  67.72621454,  67.50748676,
            67.73952427,  67.48962016,  67.42326796,  67.76330011,
            67.52701035,  67.01792733,  66.99424207,  67.56274694,
            67.42443909,  67.55925349,  67.79285796,  67.79542241,
            67.68607105,  67.11300935,  67.67777174,  67.72993186,
            67.33918455,  67.69860449,  67.13698112,  67.28325764,
            67.46150491,  67.42825846,  66.93853969,  66.94501142,
            67.674618  ,  67.52613378,  67.51379838,  67.76065074,
            67.11506578,  67.54995837,  67.51379195,  67.43352184,
            67.14158378,  67.33359402,  67.640891  ,  67.65412257,
            67.26273485,  67.66018079,  67.59257637,  67.55906596,
            67.53662031,  67.57869466,  67.41333565,  67.53192443,
            67.66909498,  67.60059629,  67.67476778,  67.29658246,
            67.59379834,  67.62628761,  67.4366966 ,  67.38991289,
            67.05532434,  67.32839781,  67.52339089,  67.4814972 ,
            67.61680572,  67.5842589 ,  67.78647861,  67.21290311,
            67.77571177,  67.55412426,  67.59177463,  67.56476222,
            67.15030151,  67.74160367,  67.44924929,  67.50222499,
            67.48111074,  67.70100821,  67.7716321 ,  67.78771736] 
         
        target_policy = [0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
            1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 1]
         
        for x,y in zip(target_value, value):
            self.assertAlmostEquals(x,y)
        for x,y in zip(target_policy, policy):
            self.assertAlmostEquals(x,y)
        self.assertEqual(iterations,4)
        self.assertAlmostEqual(residual,0)
        
        

from raam.features import GridAggregation
class TestAggregation(unittest.TestCase):
    
    def test_classify(self):
        g = GridAggregation( ((-2,+2),(-6,+6)), (10,10) )
        
        self.assertEquals(g.classify( (-2,-6) ), 0)
        self.assertEquals(g.classify( (-1.6,-6) ), 10)
        self.assertEquals(g.classify( (-2,-4) ), 1)
        self.assertEquals(g.classify( (-3,-8) ), 0)
        self.assertEquals(g.classify( (2,6) ), len(g) - 1)
        self.assertEquals(g.classify( (1.999,5.999) , range_error=True), len(g) - 1)
        
        with self.assertRaises(ValueError):
            g.classify( (-3,-8), range_error = True ) 
        with self.assertRaises(ValueError):
            g.classify( (-5,0), range_error = True ) 
        with self.assertRaises(ValueError):
            g.classify( (10,0), range_error = True ) 
        with self.assertRaises(ValueError):
            g.classify( (0,10), range_error = True ) 
        with self.assertRaises(ValueError):
            g.classify( (0,-10), range_error = True ) 
        with self.assertRaises(ValueError):
            g.classify( (2,6), range_error = True  )
    
    def test_sampling(self):
        g = GridAggregation( ((-2,+2),(-6,+6)), (10,10) )
        t = g.sampling_uniform((100,100))
        self.assertEquals(len(t), 10000)
        self.assertEquals(t[0], (-2.0, -6.0))
        self.assertEquals(t[-1], (2.0, 6.0))
        self.assertEquals(t[100], (-1.9595959595959596, -6.0))
        
        t = g.sampling_uniform((100,100), limits=((0,2),(-6,+6)))
        self.assertEquals(len(t), 10000)
        self.assertEquals(t[0], (0.0, -6.0))
        self.assertEquals(t[-1], (2.0, 6.0))
        self.assertEquals(t[100], (0.020202020202020204, -6.0))
        
    def test_eval_2d(self):
        g = GridAggregation( ((-2,+2),(-6,+6)), (10,10) )
        X,Y = g.meshgrid((50,50))
        
        # test a 0 function in 2d 
        Z = g.eval_function(zip(X.flat, Y.flat), np.zeros(100))
        Z = np.array(Z).reshape(X.shape)
        self.assertTrue(np.all(Z == np.zeros((50,50))) )
        
        # test a sin function
        g = GridAggregation( ((-2,+2),(-6,+6)), (50,50) )
        X,Y = g.meshgrid((50,50))
        Zt = np.sin(X) + np.cos(Y)
        # compute the aggregation function
        f = np.zeros(len(g))
        t = g.sampling_uniform((50,50))
        for it in t:
            f[g.classify(it)] = np.sin(it[0]) + np.cos(it[1])
        Z = g.eval_function(zip(X.flat, Y.flat), f)
        Z = np.array(Z).reshape(X.shape)
        
        self.assertTrue(np.all(Z == Zt))
        
    def test_eval_1d(self):
        g = GridAggregation( ((-6,+6),), (10,) )
        X = g.meshgrid((50,))
        
        # test a 0 function in 1d 
        Z = g.eval_function(X, np.zeros(10))
        Z = np.array(Z).reshape(X.shape)
        self.assertTrue(np.all(Z == np.zeros((50,50))))
        
        # test a sin function in 1d
        g = GridAggregation( ((-6,+6),), (50,) )
        X = g.meshgrid((50,))
        Zt = np.sin(X) 
            # compute the function for the aggregation
        f = np.zeros(len(g))
        t = g.sampling_uniform((50,))
        for it in t:
            f[g.classify(it)] = np.sin(it[0]) 
        Z = g.eval_function(X, f)
        Z = np.array(Z).reshape(X.shape)
            
        self.assertTrue(np.all(Z == Zt))

    def test_extent(self):
        g = GridAggregation( ((-2,+2),(-6,+6)), (10,15) )
        
        for i in range(len(g)):
            self.assertEquals(g.classify([x[0] for x in g.extent(i)]),i)
            
        for i in range(len(g)):
            self.assertEquals(g.classify([(x[0] + x[1]) / 2 for x in g.extent(i)]),i)

    def test_1d(self):
        g = GridAggregation( ((-1,1),), (4,))
        
        self.assertEqual(g.classify(-1), 0)
        self.assertEqual(g.classify((-1,)), 0)
        self.assertEqual(g.classify(-0.5), 1)
        self.assertEqual(g.classify(-0.50001), 0)
        self.assertEqual(g.classify(-0.001), 1)
        self.assertEqual(g.classify(0.0), 2)
        self.assertEqual(g.classify(0.49), 2)
        self.assertEqual(g.classify(0.5), 3)
        self.assertEqual(g.classify(0.99), 3)
        
    def test_all_generated(self):        
        g = GridAggregation( ((0,5),(-1,4)), (5,5) )
        
        aggs = set()
        
        for x in np.linspace(0,5,20):
            for y in np.linspace(-1,4,20):
                aggs.add(g.classify( (x,y) ) )
                
        self.assertEqual(aggs, set(range(25)))


class TestImplementable(unittest.TestCase):

    def test_construction(self):
        initial = np.ones(2) / 2
        observations = np.zeros(2,dtype=int)
        mdp = craam.MDP(2,0.99)
        mdp.add_transition(0,0,1,1.0,1.0);
        mdp.add_transition(1,0,0,1.0,1.0);

        mdpi = craam.MDPIR(mdp,observations,initial)

        rmdp = mdpi.get_robust()
        self.assertEqual(rmdp.state_count(),1)

