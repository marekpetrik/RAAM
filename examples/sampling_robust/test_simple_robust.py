import raam
from raam import robust
from raam import crobust
from raam import features
from raam import examples

import numpy as np

discount = 0.99
samples = examples.chain.simple_samples(10,discount)

r = crobust.SRoMDP(1, discount)

statenum = lambda x: np.where(x)[0][0]
zero = lambda x: 0

dict_si = {}
def sampleindex(x):
    x = statenum(x)
    index = dict_si.get(x,0)
    dict_si[x] = index + 1
    return index
    
r.from_samples(samples, decagg_big=statenum, decagg_small=zero,
                expagg_big=statenum, expagg_small=sampleindex,
                actagg=features.IdCache())
                
r.rmdp.set_uniform_distributions(0.4)

v,_,_,_ = r.rmdp.mpi_jac_l1(1000,stype=robust.SolutionType.Robust.value)

print(r.decvalue(10, v))