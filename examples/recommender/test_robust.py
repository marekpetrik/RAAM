import raam
import raam.examples
from raam import features
from raam.examples import recommender
from raam import crobust

conf = recommender.config2
conf['recommendcount'] = 1

sim = recommender.Recommender(conf)

samples, stats = recommender.unoptimized(sim, 50, 300, 'random')

samples = sim.sample_dec_ofexp(samples)
samples = sim.sample_exp_ofdec(samples, 20)

## Create C++ Samples

from raam import crobust

discrete_samples = features.DiscreteSampleView(samples)

dms = crobust.DiscreteMemSamples()

for es in discrete_samples.expsamples():
    dms.add_exp(es)

for ds in discrete_samples.decsamples():
    dms.add_dec(ds)

for ins in discrete_samples.initialsamples():
    dms.add_initial(ins)
    
assert len(list(discrete_samples.expsamples())) == len(list(dms.expsamples()))
assert len(list(discrete_samples.decsamples())) == len(list(dms.decsamples()))
assert len(list(discrete_samples.initialsamples())) == len(list(dms.initialsamples()))

## Define sampling that goes between two decision states

class DecSampleDec:
    """
    A sample that starts in a decision state
    """
    def __init__(self, decStateFrom, action, decStateTo, probability, reward):
        self.decStateFrom = decStateFrom
        self.decStateTo = decStateTo
        self.action = action
        self.probability = probability
        self.reward = reward
        
    def map(self, decmap, actmap):
        return DecSample(decmap(self.decStateFrom), actmap(self.action), \
                decmap(self.decStateTo), self.probability, self.reward)


# build a hashmap of expectation states
expsample_map = {}
for es in samples.expsamples():
    l = expsample_map.get(es.expStateFrom, None)
    if l is None:
        l = []
        expsample_map[es.expStateFrom] = l
    l.append(es)
    
# decision to decision state transitions
# repeated entries are summed
decdecsamples = []

for ds in samples.decsamples():
    
    expmatch = expsample_map[ds.expStateTo]
    
    for es in expmatch:
        
        d = DecSampleDec(ds.decStateFrom, ds.action, es.decStateTo, 1.0, es.reward)
        decdecsamples.append(d)
        
## Construct transition matrices based on the samples

import numpy as np

allstates = {dd.decStateFrom for dd in decdecsamples} 
allstates = allstates.union({dd.decStateTo for dd in decdecsamples})
allstates = sorted(allstates)

allactions = sorted({dd.action for dd in decdecsamples})

state2id = {s:i for i,s in enumerate(allstates)}
action2id = {a:i for i,a in enumerate(allactions)}

Ps = [np.zeros((len(allstates),len(allstates))) for a in allactions]
rs = [np.zeros(len(allstates)) for a in allactions]
    
for dd in decdecsamples:
    Ps[action2id[dd.action]][state2id[dd.decStateFrom], state2id[dd.decStateTo]] += dd.probability
    rs[action2id[dd.action]][state2id[dd.decStateFrom]] += dd.probability * dd.reward
    
for i,_ in enumerate(Ps):
    N = np.diag(1/(0.1 + Ps[i].sum(1)))
    Ps[i] = N.dot(Ps[i])
    rs[i] = N.dot(rs[i])

# now the initial distribution
p0 = np.zeros(len(allstates))
for ds in samples.initialsamples():
    p0[state2id[ds]] += 1

p0 = p0 / np.sum(p0)
    
## Construct and solve an MDP version

from raam import crobust

mdp = crobust.RoMDP(len(allstates),0.99)

P = np.dstack(Ps)
r = np.vstack(rs).T
a = np.arange(len(Ps),dtype=np.long)
o = np.zeros(len(Ps),dtype=np.long)

mdp.from_matrices(P,r,a,o)

sol = mdp.mpi_jac(100)

v = sol[0]
pol = sol[1]

v0 = np.linalg.solve(np.eye(19) - 0.99*Ps[0],rs[0])

print("Unoptimized:", v0.dot(p0))
print("Optimized:", v.dot(p0))
print("Improvement: %.2f%%"  % ((v.dot(p0) - v0.dot(p0))/v0.dot(p0)*100))

## Construct and solve the constrained policy

observations = np.array([s[0] for s in allstates],dtype=np.long)

mdpi = crobust.MDPIR(mdp,observations,p0)
mdpi.to_csv(b"mdp.csv",b"observ.csv",b"initial.csv",True)

mdpi.solve_reweighted(10,0.99)
