import raam
import raam. examples

from raam.examples import recommender

conf = recommender.config2
conf['recommendcount'] = 1

sim = recommender.Recommender(conf)

samples, stats = recommender.unoptimized(sim, 10, 50, 'random')

samples = sim.sample_dec_ofexp(samples)
samples = sim.sample_exp_ofdec(samples, 40)

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
        
        d = DecSampleDec(ds.decStateFrom, ds.action, es.decStateTo, 1.0 / len(expmatch), es.reward)
        decdecsamples.append(d)