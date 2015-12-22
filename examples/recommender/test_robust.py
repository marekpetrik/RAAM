import raam
import raam.examples
from raam import features
from raam.examples import recommender
from raam import crobust
import numpy as np

conf = recommender.config2
conf['recommendcount'] = 1

sim = recommender.Recommender(conf)

samples, stats = recommender.unoptimized(sim, 50, 300, 'random')

samples = sim.sample_dec_ofexp(samples)
samples = sim.sample_exp_ofdec(samples, 20)

## Create C++ Samples

from raam import crobust

discrete_samples = features.DiscreteSampleView(samples,actmapinit=sim.action_list)

dms = crobust.DiscreteMemSamples()
dms.copy_from(discrete_samples)

assert len(list(discrete_samples.expsamples())) == len(list(dms.expsamples()))
assert len(list(discrete_samples.decsamples())) == len(list(dms.decsamples()))
assert len(list(discrete_samples.initialsamples())) == len(list(dms.initialsamples()))


## Construct the sampled MDP

smdp = crobust.SMDP()

smdp.copy_samples(dms)

mdp = smdp.get_mdp(0.9)
p0 = smdp.get_initial()


## Construct and solve an MDP version

from raam import crobust

sol = mdp.mpi_jac(100)

v = sol[0]
pol = sol[1]

#v0 = np.linalg.solve(np.eye(19) - 0.99*Ps[0],rs[0])
#print("Unoptimized:", v0.dot(p0))
print("Optimized:", v.dot(p0))
#print("Improvement: %.2f%%"  % ((v.dot(p0) - v0.dot(p0))/v0.dot(p0)*100))

## Construct and solve the constrained policy

from operator import itemgetter

observations = np.array([s[0] for s in discrete_samples.all_decstates()],dtype=np.long)

mdpi = crobust.MDPIR(mdp,observations,p0)
mdpi.to_csv(b"mdp.csv",b"observ.csv",b"initial.csv",True)

#sol = mdpi.solve_reweighted(10,0.99)
