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

from raam import features
from raam import crobust
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

class BasicTests(unittest.TestCase):
    """ Tests basics """
    def test_representation(self):
        q = raam.create_test_sample()
        s = q.validate()
        r = q.statistics(0.95)['mean_return']
        self.assertAlmostEqual(5.6075, r)
        self.assertEqual(3, s['decStates'])
        self.assertEqual(3, s['expStates'])

@unittest.skipUnless(settings['opl'], 'no oplrun')
class TestOplEncoding(unittest.TestCase):
    """ Various OPL tests """

    def test_opl_encoding(self):
        samples = create_test_sample()
        generatedID = str(random.random())
        opl_data = opl.create_opl_data_v(samples, \
                 oargs={'problemName':'Test','generatedID':generatedID, \
                 'lower':0, 'discount':0.9, 'tau':10})

    def test_opl_run_alp(self):
        samples = create_test_sample()
        coeffs,solution = opl.run_opl(samples, 'Test', algorithm='ALP-v',\
                oargs={'lower':0,'upper':1000,'discount':0.9})

        self.assertEqual(solution['ProblemName'], 'Test')
        self.assertEqual(solution['Algorithm'], 'ALP-v')
        self.assertEqual(solution['Objective'], 10)
        self.assertEqual(solution['DecStateCx'], [0,10])
        self.assertEqual(solution['DecStateCx'], coeffs)

    def test_opl_run_abp(self):
        samples = create_test_sample()
        coeffs,solution = opl.run_opl(samples, 'Test', algorithm='ABP-v', \
                oargs={'lower':0,'upper':1000,'discount':0.9,'tau':10})

        self.assertEqual(solution['ProblemName'], 'Test')
        self.assertEqual(solution['Algorithm'], 'ABP-v')
        self.assertEqual(solution['Objective'], 0)
        self.assertEqual(solution['DecStateCx'], [0,10])
        self.assertEqual(solution['DecStateCx'], coeffs)

    def test_opl_run_dradp(self):
        samples = create_test_sample()
        coeffs,solution = opl.run_opl(samples, 'Test', algorithm='DRADP-v', \
                oargs={'lower':0,'upper':10,'discount':0.9,'tau':10})

        self.assertEqual(solution['ProblemName'], 'Test')
        self.assertEqual(solution['Algorithm'], 'DRADP-v')
        self.assertAlmostEqual(solution['Objective'], 10)
        self.assertEqual(solution['DecStateCx'], [-10,10])
        self.assertEqual(solution['DecStateCx'], coeffs)

class _Args:
    """ Helper class """
    pass

@unittest.skip("functionality temporarily removed")
class TestHolisticAlgorithmRuns(unittest.TestCase):
    """ Holistic tests """
    
    @unittest.skipUnless(settings['opl'], 'no oplrun')
    def test_pendulum_noextension(self):
        from raam import execute
        np.random.seed(0)
        #sp.random.seed(0)
        args = _Args()
        args.all = False
        args.problem = 'raam.examples.pendulum.Simulator()'
        args.horizon = 100
        args.extend_exp = 0
        args.extend_dec = 0
        args.runs_test = 5
        args.runs_train = 33
        args.features = 'raam.examples.pendulum.Features.gaussian'
    
        # the desired result of the algorithm
        desired = { 'dradp-opl-v':(-98.50393384519067, 0.0000), \
                    'dradp-opl-q':(0.00000,-0.95112), \
                    'alp-opl-v':(0.0000,-0.93779), \
                    'abp-opl-v':(1.0,-0.95873),\
                    'myopic' : (0.,-0.953026709441)}

        args.dradp_opl_v = True
        args.dradp_opl_q = True
        args.alp_opl_v = True
        args.abp_opl_v = True
        args.myopic = True
        args.dradp_iter_v = False
        args.default = False

        excluded = {'dradp-iter-v'} # Numerical issues 
        
        computed,_ = execute(args)

        for a,obj,mr,_ in computed:
            if a in excluded:
                continue
            self.assertAlmostEqual(mr, desired[a][0],3)
            self.assertAlmostEqual(obj, desired[a][1],3)

class TestSampleView(unittest.TestCase):
    """ Test the SampleView implementation """

    def test_samples_view(self):
        np.random.seed(0)
        random.seed(0)
        import copy
        horizon = 200 
        runs = 500
        config = copy.copy(raam.examples.recommender.config2)
        config['recommendcount'] = 1
        config['objective'] = 'margin'
        simulator = raam.examples.recommender.Recommender(config)
    
        policy_random = simulator.random_policy()
        samples_original = raam.samples.MemSamples()
        
        def toj(o):
            return json.dumps(o)
        def ofj(o):
            a = json.loads(o)
            if type(a) == list and len(a) >= 3 and type(a[2]) == list:
                a[2] = tuple(a[2])
            return a if type(a) != list else tuple(a) 
        
        samples_random = raam.samples.SampleView(samples_original,\
            decmap=ofj,decmapinv=toj,expmap=ofj,expmapinv=toj,\
            actmap=ofj,actmapinv=toj)
        
        simulator.simulate(horizon,policy_random,runs,samples=samples_random)
        
        decagg = robust.__ind__
        expagg = features.IdCache()
        self.result = robust.matrices(samples_random,decagg=decagg,expagg=expagg)
        result = self.result
        
        self.statecount = self.result['dectoexp'].shape[0]
        
        self.rmdp = crobust.RoMDP(self.statecount, 1)
        self.rmdp.from_sample_matrices(result['dectoexp'], result['exptodec'], result['actions'], result['rewards'])
        
        self.v_robust = [ 5.20241692,  3.48411739,  6.29607251,  6.29607251, 3.7897281, 6.94864048, 0.        ]
        self.v_robust_half = [  7.86077137,   9.05926647,   9.43518508,  12.21934149,  5.98868032,  13.08652546,   0.        ]

        # TODO: the actual test is missing here move this code to robust recommender

class BasicTestsSimulation(unittest.TestCase):
    """ Simulation tests """

    def setUp(self):
        self.horizon = 30

    def test_check_simulation(self):
        random.seed(1000)
        np.random.seed(1000)
        sim = raam.examples.shaping.Simulator()
        result = sim.simulate(1,raam.examples.shaping.policyNoAction,1)
        ret = result.statistics(sim.discount)['mean_return']
        stats = result.validate()
        self.assertEqual(1, stats['decStates'])
        self.assertEqual(1, stats['expStates'])
        self.assertAlmostEqual(0.040709624769089126, ret)

    def test_check_simulation_multiple(self):
        random.seed(1000)
        np.random.seed(1000)
        sim = raam.examples.shaping.Simulator()
        result = sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,range(5))
        ret = result.statistics(sim.discount)['mean_return']
        stats = result.validate()
        self.assertAlmostEqual(2.9996420337426062, ret, 4)
        self.assertEqual(30*5, stats['decStates'])
        self.assertEqual(30*5, stats['expStates'])
        
    def test_check_simulation_multiple_counter(self):
        random.seed(1000)
        np.random.seed(1000)
        horizon = 30
        sim = examples.counter.Counter()
        result = sim.simulate(horizon,sim.random_policy(),5)
        ret = result.statistics(sim.discount)['mean_return']
        self.assertAlmostEqual(-9.480222759759355, ret, 4)
        stats = result.validate()
        self.assertEqual(30*5, stats['decStates'])
        self.assertEqual(30*5, stats['expStates'])

    def test_check_simulation_multiple_counter_stateful(self):
        random.seed(1000)
        np.random.seed(1000)
        horizon = 30
        sim = examples.counter.StatefulCounter()
        result = sim.simulate(horizon,sim.random_policy(),5)
        ret = result.statistics(sim.discount)['mean_return']
        self.assertAlmostEqual(-9.480222759759355, ret, 4)
        stats = result.validate()
        self.assertEqual(30*5, stats['decStates'])
        self.assertEqual(30*5, stats['expStates'])
        
    def test_check_simulation_merge(self):
        sim = raam.examples.shaping.Simulator()
        result1 = sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,1)
        result2 = sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,[2])
        stats1 = result1.validate()
        self.assertEqual(1, stats1['runs'])
        self.assertEqual(30, stats1['decStates'])
        self.assertEqual(30, stats1['expStates'])
        stats2 = result2.validate()
        self.assertEqual(1, stats2['runs'])
        self.assertEqual(30, stats2['decStates'])
        self.assertEqual(30, stats2['expStates'])
        result = raam.samples.MemSamples()
        result.merge(result1)
        result.merge(result2)
        stats = result.validate()
        self.assertEqual(2, stats['runs'])
        self.assertEqual(60, stats['decStates'])
        self.assertEqual(60, stats['expStates'])

    def test_generate_exp_samples_ofexp(self):
        sim = raam.examples.shaping.Simulator()
        result = sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,1)
        extended = sim.sample_exp_ofexp(result,10)
        stats = extended.validate()
        self.assertEqual(30, stats['decStates'])
        self.assertEqual(330, stats['expStates'])

    def test_generate_exp_samples_ofdec(self):
        sim = raam.examples.shaping.Simulator()
        result = sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,1)
        extended = sim.sample_exp_ofdec(result,10)
        stats = extended.validate()
        self.assertEqual(30, stats['decStates'])
        self.assertEqual(330, stats['expStates'])

    def test_generate_dec_samples_ofexp(self):
        sim2 = examples.pendulum.Simulator2()
        result = sim2.simulate(30,sim2.random_policy(),1)
        extended = sim2.sample_dec_ofexp(result)
        stats = extended.validate()
        # this one does not check for duplicates
        self.assertEqual(120, stats['decStates'])
        self.assertEqual(30, stats['expStates'])

    def test_generate_dec_samples_ofdec(self):
        sim2 = examples.pendulum.Simulator2()
        result = sim2.simulate(30,sim2.random_policy(),1)
        extended = sim2.sample_dec_ofdec(result)
        stats = extended.validate()
        self.assertEqual(90, stats['decStates'])
        self.assertEqual(30, stats['expStates'])

    def test_generate_exp_samples_ofexp_replace(self):
        sim = raam.examples.shaping.Simulator()
        result = sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,1)
        extended = sim.sample_exp_ofexp(result,10,append=False)
        stats = extended.validate()
        self.assertEqual(30, stats['decStates'])
        self.assertEqual(300, stats['expStates'])

    def test_generate_exp_samples_ofdec_replace(self):
        sim = raam.examples.shaping.Simulator()
        result = sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,1)
        extended = sim.sample_exp_ofdec(result,10,append=False)
        stats = extended.validate()
        self.assertEqual(30, stats['decStates'])
        self.assertEqual(300, stats['expStates'])

    def test_generate_dec_samples_ofexp_replace(self):
        sim2 = examples.pendulum.Simulator2()
        result = sim2.simulate(30,sim2.random_policy(),1)
        extended = sim2.sample_dec_ofexp(result,append=False)
        stats = extended.validate()
        self.assertEqual(90, stats['decStates'])
        self.assertEqual(30, stats['expStates'])

    def test_generate_dec_samples_ofdec_replace(self):
        sim2 = examples.pendulum.Simulator2()
        result = sim2.simulate(30,sim2.random_policy(),1)
        extended = sim2.sample_dec_ofdec(result,append=False)
        stats = extended.validate()
        self.assertEqual(90, stats['decStates'])
        self.assertEqual(30, stats['expStates'])

    def test_greedy_policy(self):
        sim = raam.examples.shaping.Simulator()
        actions = list(itertools.permutations(range(len(sim.config.action_types)),1)) + [set()]
        greedy = sim.greedy_policy_v(lambda x:0, 10)
        result = sim.simulate(self.horizon,greedy,1)
        stats = result.validate()

        self.assertEqual(30, stats['decStates'])
        self.assertEqual(30, stats['expStates'])

    def test_greedy_better(self):
        sim = raam.examples.shaping.Simulator()
        actions = list(itertools.permutations(range(len(sim.config.action_types)),1)) + [set()]
        greedy = sim.greedy_policy_v(lambda x:0, 10)
        runs = range(10)

        result_greedy = sim.simulate(self.horizon,greedy,runs)
        result_noaction = sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,runs)

        stats_greedy = result_greedy.validate()
        self.assertEqual(300, stats_greedy['decStates'])
        self.assertEqual(300, stats_greedy['expStates'])
        stats_noaction = result_noaction.validate()
        self.assertEqual(300, stats_noaction['decStates'])
        self.assertEqual(300, stats_noaction['expStates'])
        
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
        decsamples = samples.decsamples()
        decstates = [decsample.decStateFrom for decsample in decsamples]
        
        counter = Counter(decstates)
        for state, frequency in counter.items():
            self.assertTrue(abs(SimulationTerminationTests.correct_probability(state, 1-self.probterm) \
                - frequency / self.runs) < 0.02)
    
    def test_statefull(self):
        random.seed(1982)
        c = raam.examples.counter.StatefulCounter(succ_prob=1)
        samples = c.simulate(1000,  lambda s: c.actions()[0], self.runs, probterm=self.probterm)
        decsamples = samples.decsamples()
        decstates = [decsample.decStateFrom for decsample in decsamples]
        
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
        self.assertEqual(50,len(tuple(samples.decsamples()))) 
        self.assertEqual(50,len(tuple(samples.expsamples()))) 

    def test_stateless(self):
        c = raam.examples.counter.Counter(succ_prob=1)
        samples = c.simulate(self.steps,  lambda s: c.actions(s)[0], self.runs,\
                    transitionlimit=200)
        self.assertEqual(200,len(tuple(samples.decsamples()))) 
        self.assertEqual(200,len(tuple(samples.expsamples()))) 

    def test_statefull_small(self):
        c = raam.examples.counter.StatefulCounter(succ_prob=1)
        samples = c.simulate(self.steps,  lambda s: c.actions()[0], self.runs,\
                    transitionlimit=50)
        self.assertEqual(50,len(tuple(samples.decsamples()))) 
        self.assertEqual(50,len(tuple(samples.expsamples()))) 

    def test_statefull(self):
        c = raam.examples.counter.StatefulCounter(succ_prob=1)
        samples = c.simulate(self.steps,  lambda s: c.actions()[0], self.runs,\
                    transitionlimit=200)
        self.assertEqual(200,len(tuple(samples.decsamples()))) 
        self.assertEqual(200,len(tuple(samples.expsamples()))) 

       
       
@unittest.skipUnless(settings['opl'], 'no oplrun')
class OplGenerationTests(unittest.TestCase): 
    """ Generating OPL tests """
    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        self.sim = raam.examples.shaping.Simulator()
        self.horizon = 30
        self.sim2 = examples.pendulum.Simulator2()

    def test_exp_sample_generation1(self):
        random.seed(1000)
        np.random.seed(1000)
        result = self.sim.simulate(3,self.sim.random_policy(),1)
        self.assertEqual( [e.expStateFrom[0] for e in result.expsamples()], \
                [d.expStateTo[0] for d in result.decsamples()] )

        extended1 = self.sim.sample_exp_ofexp(result,10) 
        extended2 = self.sim.sample_exp_ofdec(result,10) 

        self.assertEqual( [e.expStateFrom[0] for e in extended1.expsamples()], \
                [e.expStateFrom[0] for e in extended1.expsamples()] )

    def test_exp_sample_generation2(self):
        random.seed(1000)
        np.random.seed(1000)
        result = self.sim2.simulate(3,self.sim2.random_policy(),1)
        self.assertEqual( [e.expStateFrom[0] for e in result.expsamples()], \
                [d.expStateTo[0] for d in result.decsamples()] )

        extended1 = self.sim2.sample_exp_ofexp(result,10) 
        extended2 = self.sim2.sample_exp_ofdec(result,10) 

        self.assertEqual( [e.expStateFrom[0] for e in extended1.expsamples()], \
                [e.expStateFrom[0] for e in extended1.expsamples()] )

    def test_dec_sample_generation(self):
        random.seed(1000)
        np.random.seed(1000)
        result = self.sim2.simulate(3,self.sim2.random_policy(),1)
        del result.dec_samples[0]
        del result.exp_samples[-1]
        self.assertEqual( [e.decStateTo[0] for e in result.expsamples()], \
                [d.decStateFrom[0] for d in result.decsamples()] )

        extended1 = self.sim2.sample_dec_ofexp(result) 
        extended2 = self.sim2.sample_dec_ofdec(result) 

        self.assertEqual( [e.expStateFrom[0] for e in extended1.expsamples()], \
                [e.expStateFrom[0] for e in extended1.expsamples()] )

    @unittest.skipUnless(settings['opl'], 'no oplrun')
    def test_simple_optimization_ofexp(self):
        random.seed(1000)
        np.random.seed(1000)
        #sp.random.seed(1000)
        result = self.sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,1)
        extended = self.sim.sample_exp_ofexp(result,10) 
        extended = samples.SampleView(extended, **raam.examples.shaping.Representation)
        extended = features.apply_dec(extended,raam.examples.shaping.Features.linear[0])
        stats = extended.validate()

        coeffs,solution = opl.run_opl(extended, 'Shaping Simulation', algorithm='ALP-v', \
                oargs={'lower':-100,'upper':100,'discount':0.9})
        self.assertAlmostEqual(solution['Objective'], 7.41067, 3)

    @unittest.skipUnless(settings['opl'], 'no oplrun')
    def test_simple_optimization_ofdec(self):
        random.seed(1000)
        np.random.seed(1000)
        #sp.random.seed(1000)
        result = self.sim.simulate(self.horizon,raam.examples.shaping.policyNoAction,1)
        extended = self.sim.sample_exp_ofdec(result,10) 
        extended = samples.SampleView(extended, **raam.examples.shaping.Representation)
        extended = features.apply_dec(extended,raam.examples.shaping.Features.linear[0])
        stats = extended.validate()

        coeffs,solution = opl.run_opl(extended, 'Shaping Simulation', algorithm='ALP-v', \
                oargs={'lower':-100,'upper':100,'discount':0.9})

        self.assertAlmostEqual(solution['Objective'], 7.41067, 3)

class TestFeatures(unittest.TestCase):
    """ Test feature generation """
    
    @unittest.skip("known problem")
    def test_dec_does_not_change(self):
        q = create_test_sample()
        qb = create_test_sample()
        feature = lambda state: [1,2,3,4]
        transformed = features.apply_dec(q, feature)
        self.assertEquals(q.encode_json(), qb.encode_json())
        self.assertNotEquals(q.encode_json(), transformed.encode_json())

    @unittest.skip("known problem")
    def test_exp_does_not_change(self):
        q = create_test_sample()
        qb = create_test_sample()
        feature = lambda state: [1,2,3,4]
        transformed = features.apply_exp(q, feature)
        self.assertEquals(q.encode_json(), qb.encode_json())
        self.assertNotEquals(q.encode_json(), transformed.encode_json())
    
class TestPrecise(unittest.TestCase):
    """ Test precise MDP solvers """
    def setUp(self):
        self.samples = examples.chain.simple_samples(7,1)
        self.samplessmall = examples.chain.simple_samples(3,1)
        self.samplesstoch = examples.chain.simple_samples(7,0.75)
    
    @unittest.skipUnless(settings['opl'], 'no oplrun')
    def test_alp_no_disc(self):
        coeffs,solution = opl.run_opl(self.samples, 'Chain', algorithm='ALP-v',\
                oargs={'lower':0,'upper':1000,'discount':0})
        
        self.assertAlmostEqual(solution['Objective'], 2.428571428571428, 5)
        self.assertEqual(solution['DecStateCx'], [3,3,2,1,2,3,3])

    @unittest.skipUnless(settings['opl'], 'no oplrun')
    def test_abp_no_disc(self):
        coeffs,solution = opl.run_opl(self.samples, 'Chain', algorithm='ABP-v',\
                oargs={'lower':0,'upper':1000,'discount':0,'tau':10})
        
        self.assertAlmostEqual(solution['Objective'], 0, 5)
        self.assertEqual(solution['DecStateCx'], [3,3,2,1,2,3,3])

    @unittest.skipUnless(settings['opl'], 'no oplrun')
    def test_dradp_no_disc(self):
        coeffs,solution = opl.run_opl(self.samples, 'Chain', algorithm='DRADP-v',\
                oargs={'lower':0,'upper':1000,'discount':0, 'tau':10})
        
        self.assertAlmostEqual(solution['Objective'], 2.428571428571428, 5)
        des = [3,3,2,1,2,3,3]
        for a,b in zip(des, solution['DecStateCx']):
            self.assertAlmostEqual(a,b,5)

    @unittest.skipUnless(settings['opl'], 'no oplrun')
    def test_dradp_q_no_disc(self):
        coeffs,solution = opl.run_opl(self.samplessmall, 'Chain', algtype='expectation', algorithm='DRADP-q',\
                oargs={'lower':0,'upper':1000,'discount':0,'tau':10})

        self.assertAlmostEqual(solution['Objective'], 1, 5)
        self.assertEqual(solution['ExpStateCx'], [1,1,0,0,1,1])

    @unittest.skipUnless(settings['opl'], 'no oplrun')
    def test_alp(self):
        coeffs,solution = opl.run_opl(self.samplesstoch, 'Chain', algorithm='ALP-v',\
                oargs={'lower':0,'upper':1000,'discount':0.9})
        
        self.assertAlmostEqual(solution['Objective'], 24.65648912586719, 3)
        des = [26.0562239092526, 25.41454564669827, 23.67375336867879, 22.30637803181092, 23.67375336867879, 25.41454564669827, 26.0562239025266]
        for a,b in zip(des, solution['DecStateCx']):
            self.assertAlmostEqual(a,b,3)

    @unittest.skipUnless(settings['opl'], 'no oplrun')
    def test_abp(self):
        coeffs,solution = opl.run_opl(self.samplesstoch, 'Chain', algorithm='ABP-v',\
                oargs={'lower':0,'upper':1000,'discount':0.9,'tau':10})
        
        self.assertAlmostEqual(solution['Objective'], 0, 3)
        des = [26.0562239092526, 25.41454564669827, 23.67375336867879, 22.30637803181092, 23.67375336867879, 25.41454564669827, 26.0562239025266]
        for a,b in zip(des, solution['DecStateCx']):
            self.assertAlmostEqual(a,b,3)

    @unittest.skipUnless(settings['opl'], 'no oplrun')
    def test_dradp(self):
        coeffs,solution = opl.run_from_sample_matricesopl(self.samplesstoch, 'Chain', algorithm='DRADP-v',\
                oargs={'lower':0,'upper':1000,'discount':0.9, 'tau':10})
        
        self.assertAlmostEqual(solution['Objective'], 24.65648912586719, 3)
        des = [26.0562239092526, 25.41454564669827, 23.67375336867879, 22.30637803181092, 23.67375336867879, 25.41454564669827, 26.0562239025266]
        for a,b in zip(des, solution['DecStateCx']):
            self.assertAlmostEqual(a,b,3)

    @unittest.skipUnless(settings['opl'], 'no oplrun')
    def test_dradp_q(self):
        coeffs,solution = opl.run_opl(self.samplesstoch, 'Chain', algtype='expectation', algorithm='DRADP-q',\
                oargs={'lower':0,'upper':1000,'discount':0.9, 'tau':10})
        #print(solution)
        self.assertAlmostEqual(solution['Objective'], 24.65648912586719, 3)
        des = [26.05622390925266, 25.41454564669827, 23.67375336867879, 22.30637803181092, 21.27507794197948, 23.34243390344003, 25.26746869110318, 25.26746869110318, 23.34243390344004, 21.27507794197948, 22.30637803181091, 23.67375336867879, 25.41454564669827, 26.05622390925266]
        
        for a,b in zip(des, solution['ExpStateCx']):
            self.assertAlmostEqual(a,b,4)

    @unittest.skipUnless(settings['opl'], 'no oplrun')
    def test_dradp(self):
        coeffs,solution = opl.run_opl(self.samplesstoch, 'Chain', algorithm='DRADP-v',\
                oargs={'lower':0,'upper':1000,'discount':0.9, 'tau':10})
        
        self.assertAlmostEqual(solution['Objective'], 24.65648912586719, 3)
        des = [26.0562239092526, 25.41454564669827, 23.67375336867879, 22.30637803181092, 23.67375336867879, 25.41454564669827, 26.0562239025266]
        for a,b in zip(des, solution['DecStateCx']):
            self.assertAlmostEqual(a,b,3)


    @unittest.skipUnless(settings['cvxopt'], 'no cvxopt')
    def test_dradp_iter(self):
        coeffs,solution = direct.solve_v(0.9, samples=self.samplesstoch, verbose=False)
        
        self.assertAlmostEqual(solution['Objective'], 24.65648912586719, 3)
        des = [26.0562239092526, 25.41454564669827, 23.67375336867879, 22.30637803181092, 23.67375336867879, 25.41454564669827, 26.0562239025266]
        for a,b in zip(des, solution['DecStateCx']):
            self.assertAlmostEqual(a,b,3)
            
    def test_robust_stoch(self):
        decagg = features.IndexCache()
        expagg = features.IdCache()
        result = robust.matrices(self.samplesstoch,decagg=decagg,expagg=expagg)
        values, policy, residual = robust.vi_gs(result['dectoexp'],result['exptodec'],result['rewards'],0.9,200)
    
        des = [26.0562239092526, 25.41454564669827, 23.67375336867879, 22.30637803181092, 23.67375336867879, 25.41454564669827, 26.0562239025266]
        for a,b in zip(des, values):
            self.assertAlmostEqual(a,b,3)
         
    def test_crobust_stoch(self):
        decagg = features.IndexCache()
        expagg = features.IdCache()
        result = robust.matrices(self.samplesstoch,decagg=decagg,expagg=expagg)
        rmdp = crobust.RoMDP(7,0.9)
        rmdp.from_sample_matrices(result['dectoexp'],result['exptodec'],result['actions'],result['rewards'])
        valuefunction,_,_,_,_ = rmdp.vi_gs(200)
    
        des = [26.0562239092526, 25.41454564669827, 23.67375336867879, 22.30637803181092, 23.67375336867879, 25.41454564669827, 26.0562239025266]
        for a,b in zip(des, valuefunction):
            self.assertAlmostEqual(a,b,3)     
            
    def test_crobust_stoch_re(self):
        decagg = features.IndexCache()
        expagg = features.IdCache()
        result = robust.matrices(self.samplesstoch,decagg=decagg,expagg=expagg)
        rmdp = crobust.RoMDP(7,0.9)
        rmdp.from_sample_matrices(result['dectoexp'],result['exptodec'],result['actions'],result['rewards'])
        valuefunction,_,_,_,_ = rmdp.vi_jac(200)
    
        des = [26.0562239092526, 25.41454564669827, 23.67375336867879, 22.30637803181092, 23.67375336867879, 25.41454564669827, 26.0562239025266]
        for a,b in zip(des, valuefunction):
            self.assertAlmostEqual(a,b,3)     

    def test_robust_stoch(self):
        decagg = features.IndexCache()
        expagg = features.IdCache()
        result = robust.matrices(self.samplesstoch,decagg=decagg,expagg=expagg)
        values, policy, residual = robust.vi_gs(result['dectoexp'],result['exptodec'],result['rewards'],0.9,200)
    
        des = [26.0562239092526, 25.41454564669827, 23.67375336867879, 22.30637803181092, 23.67375336867879, 25.41454564669827, 26.0562239025266]
        for a,b in zip(des, values):
            self.assertAlmostEqual(a,b,3)
            
    def test_robust_deter(self):
        decagg = features.IndexCache()
        expagg = features.IdCache()
        result = robust.matrices(self.samples,decagg=decagg,expagg=expagg)
        values, policy, residual = robust.vi_gs(result['dectoexp'],result['exptodec'],result['rewards'],0.9,300)
    
        des = [ 30., 30., 29., 27.1, 29., 30., 30.]
        for a,b in zip(des, values):
            self.assertAlmostEqual(a,b,2)

    def test_crobust_deter(self):
        decagg = features.IndexCache()
        expagg = features.IdCache()
        result = robust.matrices(self.samples,decagg=decagg,expagg=expagg)
        rmdp = crobust.RoMDP(7,0.9)
        rmdp.from_sample_matrices(result['dectoexp'],result['exptodec'],result['actions'],result['rewards'])
        valuefunction,_,_,_,_ = rmdp.vi_gs(200)
    
        des = [ 30., 30., 29., 27.1, 29., 30., 30.]
        for a,b in zip(des, valuefunction):
            self.assertAlmostEqual(a,b,2)
            
    def test_crobust_deter_re(self):
        decagg = features.IndexCache()
        expagg = features.IdCache()
        result = robust.matrices(self.samples,decagg=decagg,expagg=expagg)
        rmdp = crobust.RoMDP(7,0.9)
        rmdp.from_sample_matrices(result['dectoexp'],result['exptodec'],result['actions'],result['rewards'])
        valuefunction,_,_,_,_ = rmdp.vi_jac(200)
    
        des = [ 30., 30., 29., 27.1, 29., 30., 30.]
        for a,b in zip(des, valuefunction):
            self.assertAlmostEqual(a,b,2)            

@unittest.skip("capability removed temporarily")
class MatrixConstruction(unittest.TestCase):
    """ Test matrix construction from samples."""
    def setUp(self):
        self.samples = examples.chain.simple_samples(7,1)
        self.samplessmall = examples.chain.simple_samples(3,1)
        self.samplesstoch = examples.chain.simple_samples(7,0.75)
        self.sim = examples.pendulum.Simulator()

    def test_basic_construction(self):
        m = matrices(self.samples, lambda s:'l')
        m['dectoexp'][None] = m['dectoexp'][None].todense()
        m['exptodec'] = m['exptodec'].todense()

    def test_basic_construction_nopol(self):
        m = matrices(self.samplessmall,policy_repr='all')
        m['dectoexp']['l'] = m['dectoexp']['l'].todense()
        m['dectoexp']['r'] = m['dectoexp']['r'].todense()
        m['exptodec'] = m['exptodec']

    @unittest.skipUnless(settings['cvxopt'], 'no cvxopt')
    def test_basic_bound_compute_nodisc(self):
        def optpolicy(decstate):
            return 'l' if np.argmax(decstate) <= len(decstate)/2 else 'r'

        solution = direct.bound_samples(self.samples, 0, 'lower', optpolicy)
        self.assertAlmostEqual(solution['Objective'], 2.428571428571428, 3)
        self.assertEqual(sorted(solution['DecStateCx']), sorted([3,3,2,1,2,3,3]))

    @unittest.skipUnless(settings['cvxopt'], 'no cvxopt')
    def test_basic_bound_compute(self):
        def optpolicy(decstate):
            return 'l' if np.argmax(decstate) <= len(decstate)/2 else 'r'

        solution = direct.bound_samples(self.samplesstoch, 0.9, 'lower', optpolicy)
        self.assertAlmostEqual(solution['Objective'], 24.65648912586719, 3)
        des = [26.0562239092526, 25.41454564669827, 23.67375336867879, 22.30637803181092, 23.67375336867879, 25.41454564669827, 26.0562239025266]
        for a,b in zip(sorted(des), sorted(solution['DecStateCx'])):
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
            
    def test_matrix_construction(self):
        samples = examples.chain.simple_samples(7,1)
        decagg = features.IndexCache()
        expagg = features.IdCache()
    
        result = robust.matrices(samples,decagg=decagg,expagg=expagg)
        
        for r in result['dectoexp']:
            self.assertEqual(r.shape, (2,1))
        
        toexpstates  = np.concatenate(result['dectoexp']).flatten()
            
        self.assertEqual(len(set(toexpstates)), 14)
        self.assertEqual(len(toexpstates), 14)
        self.assertEqual(result['exptodec'].shape, (14, 7))
        self.assertEqual(result['dectoexp'].shape, (7,))
        self.assertEqual(len(result['initial']), 7)
        self.assertEqual(result['rewards'].shape, (14,))
        self.assertEqual(len(result['actions']), 2)
        
    def test_robust_deter(self):
        decagg = features.IndexCache()
        expagg = features.IdCache()
        samples = examples.chain.simple_samples(7,0.75)
        result = robust.matrices(samples,decagg=decagg,expagg=expagg)
        values, policy, residual = robust.vi_gs(result['dectoexp'],result['exptodec'],result['rewards'],0.9,300)
    
        des = [26.0562239092526, 25.41454564669827, 23.67375336867879, 22.30637803181092, 23.67375336867879, 25.41454564669827, 26.0562239025266]
        for a,b in zip(des, values):
            self.assertAlmostEqual(a,b,2)        
            
        self.assertEquals(list(policy.argmax(1)), [0, 0, 0, 0, 1, 1, 1])
        
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
        self.assertEquals(crobust.cworstcase_l1(z,q,t),q.dot(z))
        
        q = np.array([0.4, 0.3, 0.1, 0.2])
        z = np.array([1.0,2.0,5.0,4.0])
        t = 1
        self.assertAlmostEqual(crobust.cworstcase_l1(z,q,t),1.1)        
        
        q = np.array([0.4, 0.3, 0.1, 0.2])
        z = np.array([1.0,2.0,5.0,4.0])
        t = 2
        self.assertEquals(crobust.cworstcase_l1(z,q,t),np.min(z))
        
        q = np.array([1.0,0.0])
        z = np.array([2.0,1.0])
        t = 0
        self.assertEquals(crobust.cworstcase_l1(z,q,t),2)
        
        q = np.array([1.0,0.0])
        z = np.array([2.0,1.0])
        t = 1
        self.assertEquals(crobust.cworstcase_l1(z,q,t),1.5)
        
        q = np.array([1.0,0.0])
        z = np.array([2.0,1.0])
        t = 2
        self.assertEquals(crobust.cworstcase_l1(z,q,t),1)
    
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
        
        rmdp = crobust.RoMDP(states,0.99)
        rmdp.from_matrices(transitions,rewards,actions,outcomes)
        value,policy,residual,iterations,_ = rmdp.mpi_jac(1000)
         
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
        
        
from operator import itemgetter
        
class RobustRecommender(unittest.TestCase):       
    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        import copy
        horizon = 200 
        runs = 500
        config = copy.copy(raam.examples.recommender.config2)
        config['recommendcount'] = 1
        config['objective'] = 'margin'
        simulator = raam.examples.recommender.Recommender(config)
    
        policy_random = simulator.random_policy()
        samples_random = simulator.simulate(horizon,policy_random,runs)
        
        decagg = robust.__ind__
        expagg = features.IdCache()
        self.result = robust.matrices(samples_random,decagg=decagg,expagg=expagg)
        result = self.result
        
        self.statecount = self.result['dectoexp'].shape[0]
        
        self.rmdp = crobust.RoMDP(self.statecount,1.0)
        self.rmdp.from_sample_matrices(result['dectoexp'], result['exptodec'], result['actions'], result['rewards'])
        
        self.v_robust = [ 5.20241692,  3.48411739,  6.29607251,  6.29607251, 3.7897281, 6.94864048, 0.        ]
        self.v_robust_half = [  7.86077137,   9.05926647,   9.43518508,  12.21934149,  5.98868032,  13.08652546,   0.        ]
        
    def _check_vrobust(self,v):
        for iv, ivr in zip(v, self.v_robust):
            self.assertAlmostEqual(iv, ivr, 2)
            
    def _check_vrobust_half(self,v):
        for iv, ivr in zip(v, self.v_robust_half):
            self.assertAlmostEqual(iv, ivr, 2)            
        
    def test_many_iterations(self):
        v = self.rmdp.vi_gs(1000)    
        self._check_vrobust(v[0])
        
    def test_residual(self):
        v = self.rmdp.vi_gs(10000, maxresidual=0.0001)
        self._check_vrobust(v[0])
    
    def test_many_iterations_with_replace(self):
        v = self.rmdp.vi_jac(1000)
        self._check_vrobust(v[0])
        
    def test_residual_with_replace(self):
        v = self.rmdp.vi_jac(10000, maxresidual=0.0001)
        self._check_vrobust(v[0])
    
    def test_l1_worst(self):
        self.rmdp.set_uniform_thresholds(2)
        v = self.rmdp.vi_gs_l1(1000)
        self._check_vrobust(v[0])
    
    def test_l1_worst_with_residual(self):
        self.rmdp.set_uniform_thresholds(2)
        v = self.rmdp.vi_gs_l1(10000, maxresidual=0.0001)
        self._check_vrobust(v[0])
    
    def test_l1_worst_with_replace(self):
        self.rmdp.set_uniform_thresholds(2)
        v = self.rmdp.vi_jac_l1(1000)
        self._check_vrobust(v[0])
    
    def test_l1_worst_with_residual_and_replace(self):
        self.rmdp.set_uniform_thresholds(2)
        v = self.rmdp.vi_jac_l1(10000, maxresidual=0.0001)
        self._check_vrobust(v[0])
        
    def test_python_implementation(self):
        result = self.result
        values, policy_vec, residual = robust.vi_gs(result['dectoexp'],\
                                    result['exptodec'],result['rewards'],1,100)
        self._check_vrobust(values)
    
    def test_l1_half(self):
        self.rmdp.set_uniform_thresholds(0.5)
        v = self.rmdp.vi_gs_l1(1000)
        self._check_vrobust_half(v[0])
    
    def test_l1_half_with_residual(self):
        self.rmdp.set_uniform_thresholds(0.5)
        v = self.rmdp.vi_gs_l1(10000, maxresidual=0.0001)
        self._check_vrobust_half(v[0])
    
    def test_l1_half_with_replace(self):
        self.rmdp.set_uniform_thresholds(0.5)
        v = self.rmdp.vi_jac_l1(1000)
        self._check_vrobust_half(v[0])
    
    def test_l1_half_with_residual_and_replace(self):
        self.rmdp.set_uniform_thresholds(0.5)
        v = self.rmdp.vi_jac_l1(10000, maxresidual=0.0001)
        self._check_vrobust_half(v[0])        

import math

class RobustFromSamples(unittest.TestCase):

    def test_robust_raw_samples(self):
        # make sure that the value is computed correctly for a problem with
        # stochastic transitions
        import raam
        from raam import crobust

        m = raam.MemSamples()
        
        m.add_dec(raam.DecSample([0],[0],[0],0,0))
        m.add_dec(raam.DecSample([1],[0],[1],0,0))
        
        m.add_exp(raam.ExpSample([0],[0],1.0,1.0,0,0))
        m.add_exp(raam.ExpSample([0],[1],2.0,1.0,0,0))
        m.add_exp(raam.ExpSample([0],[1],3.0,1.0,0,0))
        m.add_exp(raam.ExpSample([1],[1],0.0,1.0,0,0))
        
        r = crobust.SRoMDP(2,0.9)

        l = itemgetter(0)

        r.from_samples(m,decagg_big=l,decagg_small=l,expagg_big=l,actagg=l,expagg_small=None)
        
        r.rmdp.set_uniform_distributions(1.0)
        v = r.rmdp.vi_jac(1000,stype=2)[0]
        v = r.decvalue(2,v)
        
        # compute the true value function
        P = np.array([[1/3, 2/3],[0,1]])
        import numpy.linalg as la
        vt = la.solve( (np.eye(2) - 0.9*P), np.array([2,0]) )
        
        self.assertAlmostEqual(v[0], vt[0], 5)
        self.assertAlmostEqual(v[1], vt[1], 5)


    def test_robust_add_samples(self):
        # makes sure that adding more samples sequentially works well
        # the solution should be the same regadless of the order of samples add
        # or if they are all added at the same time
        import raam
        from raam import crobust
        from operator import itemgetter

        m1 = raam.MemSamples()
        m2 = raam.MemSamples()
        
        m1.add_dec(raam.DecSample([0,0],[0],[1],0,0))
        m1.add_dec(raam.DecSample([0,0],[0],[2],0,0))
        m2.add_dec(raam.DecSample([0,0],[0],[3],0,0))
        
        # this is the worse outcome
        m2.add_dec(raam.DecSample([0,1],[0],[1],0,0))
        m2.add_dec(raam.DecSample([0,1],[0],[2],0,0))
        
        m1.add_exp(raam.ExpSample([1],[1],3,1.0,0,0))
        m1.add_exp(raam.ExpSample([1],[2],2,1.0,0,0))
        m2.add_exp(raam.ExpSample([1],[3],1,1.0,0,0))

        m2.add_exp(raam.ExpSample([2],[1],3,1.0,0,0))
        m1.add_exp(raam.ExpSample([2],[2],2,1.0,0,0))
        m1.add_exp(raam.ExpSample([2],[3],1,1.0,0,0))


        m2.add_exp(raam.ExpSample([3],[1],3,1.0,0,0))
        m1.add_exp(raam.ExpSample([3],[2],2,1.0,0,0))
        m2.add_exp(raam.ExpSample([3],[3],3,1.0,0,0))
       

        l = itemgetter(0)
        s = itemgetter(1)        
            
        # test one order
        r = crobust.SRoMDP(2,0.9)
        r.from_samples(m1,decagg_big=l,decagg_small=s,expagg_big=l,actagg=l,expagg_small=None)
        r.from_samples(m2,decagg_big=l,decagg_small=s,expagg_big=l,actagg=l,expagg_small=None)

        r.rmdp.set_uniform_distributions(0.0)
        v = r.rmdp.vi_jac(3000,stype=2)[0]

        self.assertAlmostEqual(r.decvalue(1,v)[0], 2.11111111)
        self.assertAlmostEqual(r.expvalue(4,v)[1], 2.0)
        self.assertAlmostEqual(r.expvalue(4,v)[2], 2.0)
        self.assertAlmostEqual(r.expvalue(4,v)[3], 2.66666666)

        # test merged
        m = raam.MemSamples()
        m.merge(m1)
        m.merge(m2)

        r = crobust.SRoMDP(2,0.9)
        r.from_samples(m,decagg_big=l,decagg_small=s,expagg_big=l,actagg=l,expagg_small=None)
        
        r.rmdp.set_uniform_distributions(0.0)
        v = r.rmdp.vi_jac(3000,stype=2)[0]

        self.assertAlmostEqual(r.decvalue(1,v)[0], 2.11111111)
        self.assertAlmostEqual(r.expvalue(4,v)[1], 2.0)
        self.assertAlmostEqual(r.expvalue(4,v)[2], 2.0)
        self.assertAlmostEqual(r.expvalue(4,v)[3], 2.66666666)

        # test reversed order
        r = crobust.SRoMDP(2,0.9)
        r.from_samples(m2,decagg_big=l,decagg_small=s,expagg_big=l,actagg=l,expagg_small=None)
        r.from_samples(m1,decagg_big=l,decagg_small=s,expagg_big=l,actagg=l,expagg_small=None)
        
        r.rmdp.set_uniform_distributions(0.0)
        v = r.rmdp.vi_jac(3000,stype=2)[0]

        self.assertAlmostEqual(r.decvalue(1,v)[0], 2.11111111)
        self.assertAlmostEqual(r.expvalue(4,v)[1], 2.0)
        self.assertAlmostEqual(r.expvalue(4,v)[2], 2.0)
        self.assertAlmostEqual(r.expvalue(4,v)[3], 2.66666666)

@unittest.skipUnless(settings['opl'], 'no oplrun')
class RobustRecommenderOptimistic(unittest.TestCase):       
    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        import copy
        horizon = 200 
        runs = 500
        config = copy.copy(raam.examples.recommender.config2)
        config['recommendcount'] = 1
        config['objective'] = 'margin'
        simulator = raam.examples.recommender.Recommender(config)
    
        policy_random = simulator.random_policy()
        samples_random = simulator.simulate(horizon,policy_random,runs)
        
        decagg = robust.__ind__
        expagg = features.IdCache()
        self.result = robust.matrices(samples_random,decagg=decagg,expagg=expagg)
        result = self.result
        
        self.statecount = self.result['dectoexp'].shape[0]
        
        self.rmdp = crobust.RoMDP(self.statecount, 1)
        self.rmdp.from_sample_matrices(result['dectoexp'], result['exptodec'], result['actions'], result['rewards'])
        
        self.v_robust = [ 20.,  20.,  20.,  20.,  20.,  20.,   0.]
        self.v_robust_half = [ 15.91487572,  19.0008396 ,  16.50006595,  18.91015309,  16.51552054,  19.79035588,   0.        ]
        
    def _check_vrobust(self,v):
        for iv, ivr in zip(v, self.v_robust):
            self.assertAlmostEqual(iv, ivr, 2)
            
    def _check_vrobust_half(self,v):
        for iv, ivr in zip(v, self.v_robust_half):
            self.assertAlmostEqual(iv, ivr, 2)            
        
    def test_many_iterations(self):
        v = self.rmdp.vi_gs(1000, stype=1)    
        self._check_vrobust(v[0])
        
    def test_residual(self):
        v = self.rmdp.vi_gs(10000, maxresidual=0.0001, stype=1)
        self._check_vrobust(v[0])
    
    def test_many_iterations_with_replace(self):
        v = self.rmdp.vi_jac(1000, stype=1)
        self._check_vrobust(v[0])
        
    def test_residual_with_replace(self):
        v = self.rmdp.vi_jac(10000, maxresidual=0.0001, stype=1)
        self._check_vrobust(v[0])
    
    def test_l1_worst(self):
        self.rmdp.set_uniform_thresholds(2)
        v = self.rmdp.vi_gs_l1(1000, stype=1)
        self._check_vrobust(v[0])
    
    def test_l1_worst_with_residual(self):
        self.rmdp.set_uniform_thresholds(2)
        v = self.rmdp.vi_gs_l1(10000, maxresidual=0.0001, stype=1)
        self._check_vrobust(v[0])
    
    def test_l1_worst_with_replace(self):
        self.rmdp.set_uniform_thresholds(2)
        v = self.rmdp.vi_jac_l1(1000, stype=1)
        self._check_vrobust(v[0])
    
    def test_l1_worst_with_residual_and_replace(self):
        self.rmdp.set_uniform_thresholds(2)
        v = self.rmdp.vi_jac_l1(10000, maxresidual=0.0001, stype=1)
        self._check_vrobust(v[0])
        
    def test_python_implementation(self):
        result = self.result
        values, policy_vec, residual, _ = robust.vi_gs(result['dectoexp'],result['exptodec'],result['rewards'],1,100,type='optimistic')
        self._check_vrobust(values)
    
    def test_l1_half(self):
        self.rmdp.set_uniform_thresholds(0.5)
        v = self.rmdp.vi_gs_l1(1000, stype=1)
        self._check_vrobust_half(v[0])
    
    def test_l1_half_with_residual(self):
        self.rmdp.set_uniform_thresholds(0.5)
        v = self.rmdp.vi_gs_l1(10000, maxresidual=0.0001, stype=1)
        self._check_vrobust_half(v[0])
    
    def test_l1_half_with_replace(self):
        self.rmdp.set_uniform_thresholds(0.5)
        v = self.rmdp.vi_jac_l1(1000, stype=1)
        self._check_vrobust_half(v[0])
    
    def test_l1_half_with_residual_and_replace(self):
        self.rmdp.set_uniform_thresholds(0.5)
        v = self.rmdp.vi_jac_l1(10000, maxresidual=0.0001, stype=1)
        self._check_vrobust_half(v[0])      
    

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
        observations = np.zeros(2)
        mdp = crobust.RoMDP(2,0.99)
        mdp.add_transition_d(0,0,1,1.0,1.0);
        mdp.add_transition_d(1,0,0,1.0,1.0);

        mdpi = crobust.MDPIR(mdp,observations,initial)

        rmdp = mdpi.get_robust()
        self.assertEqual(rmdp.state_count(),1)
        
        
