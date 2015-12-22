"""
=================================================
Approximation features (:mod:`raam.features`)
=================================================

Provides methods for conveniently applying features and also a set of
commonly used feature functions.
"""
import numpy as np
import math
import copy
import raam.samples
from operator import itemgetter


def apply_dec(samples,features):
    """
    Returns new samples with the given features applied to decision states
    
    Parameters
    ----------
    samples :` raam.Samples
        Source of the samples
    features : function (list -> list)
        Function that computes feature values

    Returns
    -------
    out : raam.Samples
        Samples after the applied feature map
    """
    nsamples = raam.samples.MemSamples()
    
    for d in samples.decsamples():
        dd = copy.copy(d)
        dd.decStateFrom = features(dd.decStateFrom) 
        nsamples.add_dec(dd)

    for e in samples.expsamples():
        ee = copy.copy(e)
        ee.decStateTo = features(e.decStateTo) 
        nsamples.add_exp(ee)

    return nsamples

def apply_exp(samples,features):
    """
    Returns new samples with the given features applied to expectation states

    Parameters
    ----------
    samples : raam.Samples
        Source of the samples
    features : function (list -> list)
        Function that computes feature values

    Returns
    -------
    out : raam.Samples
        Samples after the applied feature map
    """
    nsamples = raam.samples.MemSamples()
    
    for d in samples.decsamples():
        dd = copy.copy(d)
        dd.expStateTo = features(d.expStateTo) 
        nsamples.add_dec(dd)

    expStates = []
    for e in samples.expsamples():
        ee = copy.copy(e)
        ee.expStateFrom = features(e.expStateFrom) 
        nsamples.add_exp(ee)
    
    return nsamples

def gaussian(centerpoints):
    """ Gaussian features """
    return (lambda p: [ math.exp(- np.linalg.norm(np.array(p)-px) ** 2 ) for px in centerpoints ] + [1] )

def gaussian_action(centerpoints, actions):
    """ Gaussian features, with the last one being the action """

    def funct(p):
        result = []
        action = p[-1]
        feats = [ math.exp(- np.linalg.norm(np.array(p[:-1])-px) ** 2 ) for px in centerpoints ]
        zeros = [0] * len(feats)
        for a in actions: 
            if a == action: result.extend(feats)
            else: result.extend(zeros)
        result.append(1)
        return result

    return funct

def piecewise_linear(bp):
    """ Piecewise linear approximation with customizable points of nonlinearity """

    if bp is not None:
        assert False, 'Not implemented'

    return (lambda p: p + [1])


import bisect
from operator import mul
from functools import reduce
from itertools import product

class GridAggregation:
    """
    Class that handles state aggregation. Each aggregate state is *closed* on the
    lower bound and *open* on the upper bound.
    
    The first dimension is the most significant one
    
    Use ``len(x)'' to determine the number of states

    Examples
    --------
    A two-dimensional grid can be constructed as:
    
    >>> GridAggregation( ((-2,+2),(-6,+6)), (10,10) )
    
    A one-dimensional grid can be constructed as:
    
    >>> GridAggregation( ((-6,+6),), (10,) )
    """

    def __init__(self, limits, ticks):
        """
        Initializes the grid
        
        Parameters
        ----------
        limits : list
            List of min and max limits for each dimension
        ticks : list
            Resolution of each dimension
        
        """
        # the spacing for each dimension, includes both the first and the last points
        if len(ticks) != len(limits):
            raise ValueError('Limits and ticks must have the same length.')
        self._limits = limits
        self._grids = tuple(np.linspace(d[0],d[1],t+1) for t,d in zip(ticks,limits) )
        self._length = reduce(mul, (len(g) - 1 for g in self._grids))
    
    
    def classify(self,state,range_error=False):
        """
        Computes the appropriate class for the grid clustering
        
        Parameters
        ----------
        state : array
            N-dimensional array representing the sample. Also can process a single
            number.
        range_error : bool
            Whether an error should be raised when the state is outside of 
            the limits. Otherwise all states are mapped to the boundaries.
        
        Returns
        -------
        out : int
            The number of cluster in which the state belongs
        """

        if '__len__' not in dir(state):
            # this is meant to address the 1-dimensional case
            state = (state,)
        
        if len(state) != len(self._grids):
            raise ValueError('Incorrect dimensions of the point. Must match the number of grid dimensions.')
        
        stateindex = 0
        for i,g in zip(state,self._grids):
            ind = bisect.bisect(g,i) -1
            if ind >= len(g) - 1 or ind < 0:
                if range_error:
                    raise ValueError('Index value %s is outside of the bounds for state: %s.' % (str(i),str(state)))
                else:
                    ind = min(max(ind,0),len(g)-2)
            
            stateindex = stateindex * (len(g) - 1) + ind
            
        return stateindex
    
    def extent(self, gridno):
        """
        Returns the extent of the grid (a single aggregation) for all dimensions
        
        Parameters
        ----------
        gridno : int
            Number of the grid
        """
        if gridno < 0 or gridno >= len(self):
            raise ValueError('Invalid grid number. Must be smaller than the length of the aggregation.')
        
        grids = self._grids
        result = [None] * len(grids)
        
        for i in range(len(grids)-1,-1,-1):
            g = grids[i]
            index = gridno % (len(g)-1)
            gridno = gridno // (len(g)-1)
            
            result[i] = (g[index],g[index+1])
        
        return result
    
    def sampling_uniform(self,ticks,limits=None):
        """
        Generates a set of uniform samples for the given limits
        
        Parameters
        ----------
        ticks : array
            Resolutions in each dimension
        limits : array, optional
            The limits of sampling. If not provided, then the limits of the 
            aggregation are used
            
        Returns
        -------
        out : iterator
            A product iterators that consists of arrays of states
        """
        
        if limits is None:
            limits = self._limits

        if len(ticks) != len(self._limits) or len(limits) != len(self._limits):
            raise ValueError('The number of ticks or limit entries must match the dimensions of the grid.')
            
        spaces = [np.linspace(l[0],l[1],t) for l,t in zip(limits,ticks)]
        return list(product(*spaces))
        
    def eval_function(self,x,y):
        """
        Evaluates a function defined on the aggregation on a set of points 
        
        Parameters
        ----------
        x : list, array
            An array of values that correspond to the dimension of the aggregation
        y : list, array with length equal to the number of aggregates
        
        Returns
        -------
        out : list, array with length matching length of x
            List of values that correspond to the function y values on the set
            of points defined by x
        """
        if len(y) != len(self):
            raise ValueError('The number of function values must match the number of aggregate regions.')

        return [y[self.classify(ix,range_error=False)] for ix in x]
    
    def meshgrid(self,ticks,limits=None):
        """
        Constructs a meshgrid (like numpy.meshgrid) on the aggregation object
        
        Parameters
        ----------
        ticks : list
            Resolution along each direction (number of points including the 
                limit points)
        limits : array, optional
            The limits of sampling. If not provided, then the limits of the 
            aggregation are used
            
        Returns
        -------
        X1, .. Xn : numpy.ndarray, n is the number of tick dimensions
            Arrays with the same dimension and the number of tick dimensions with
            the size corresponding to the number of ticks along that dimension
        """
        if limits is None:
            limits = self._limits
        
        if len(ticks) != len(self._limits) or len(limits) != len(self._limits):
            raise ValueError('The number of ticks or limit entries must match the dimensions of the grid.')
       
        spaces = [np.linspace(l[0],l[1],t) for l,t in zip(limits,ticks)]
        if len(limits) >= 2:
            return np.meshgrid(*spaces)
        else:
            return spaces[0]
        
    def __len__(self):
        """The number of aggregated states"""
        return self._length

class IdCache:
    """
    Creates and caches id's of list objects. When an id of an object that is 
    not cached, then a new id is created. The ids are retrieved by directly 
    calling the class instance/object as a function (or use the method __call__). 
    This class can be used to generate id's online for a set of values.
    
    The list is converted to a tuple.
    
    Example
    -------
    Creating a cache 
    >>> cache = IdCache()
    Request new object ids
    >>> cache([1,2,3])
    >>> cache([3,4,5])
    Request and id of an object that is already cached
    >>> cache([3,4,5])
    
    See Also
    --------
    IndexCache
    """
    def __init__(self):
        self.state_vals = {}
        
    def __call__(self,state,add=True):
        """
        Adds an object to the cache of object id. Return an existing id if the 
        object already exists.
        
        Parameters
        ----------
        add : bool, optional
            Whether to add the element to the cache is not present
        """
        ts = tuple(state)
        if ts in self.state_vals:
            return self.state_vals[ts]
        elif add:
            nn = len(self.state_vals) 
            self.state_vals[ts] = nn
            return nn
        else:
            raise ValueError('Element not present in the cache and add=False.')


class IndexCache:
    """
    Creates and caches id's of list objects within each aggregation. Based on
    :mod:`IdCache` with each id being specific to the actual aggregate state.
    The id's are retrieved by simply calling the object as a function (or
    through __call__). The parameters are assumed to be convertible to tuples.
    
    
    Example
    -------
    Creating a cache 
    >>> cache = IndexCache()
    Request new object ids
    >>> cache([1,2,3])
    >>> cache([3,4,5])
    Request and id of an object that is already cached
    >>> cache([3,4,5])
    
    See Also
    --------
    IdCache
    """
    def __init__(self,aggregation=None):
        self.index_vals = {}

        if aggregation is None:
            aggregation = IdCache()

        if not callable(aggregation):
            raise ValueError('The aggregation parameter must be a function.')
        self.idcache = aggregation
        
    def __call__(self,state,add=True):
        """ 
        Returns a tuple of number of aggregation state and the index within 
        the aggregation.
        
        Parameters
        ----------
        add : bool, optional
            Whether to add the element to the cache is not present
        """
        agg = self.idcache(state)
        
        if agg not in self.index_vals:
            if add:
                self.index_vals[agg] = IdCache()
            else:
                raise ValueError('Element not present in the cache and add=False.')
        index = self.index_vals[agg](state,add=add)
        return (agg, index)


class DiscreteSampleView(raam.samples.SampleView):
    """
    Creates a sample view which assigns discrete numbers to decision states,
    expectation states, and actions. 
    
    State ids are constructed on demand.
    
    This class can be also used to reduce the number of states if they are
    not generated contigously, such as when they come from GridAggregation
    
    Arguments
    ---------
    decmap : IdCache
        Maps decision states to indexes
    expmap : IdCache
        Maps expectation states to indexes
    actmap : IdCache
        Maps actions to indexes
        
    Parameters
    ----------
    samples : Samples
        Samples to map state values to indexes
    decmapinit : list 
        List of values to be assigned ids beginning with 0 to init decision state numbers
    expmapinit : list 
        List of values to be assigned ids beginning with 0 to init expectation state numbers
    actmapinit : list 
        List of values to be assigned ids beginning with 0 to init action numbers
    """
    
    def __init__(self,samples,decmapinit=[],expmapinit=[],actmapinit=[]):
        self.decmap = IdCache()
        for v in decmapinit:
            self.decmap(v)
        
        self.expmap = IdCache()
        for v in expmapinit:
            self.expmap(v)        
        
        self.actmap = IdCache()
        for v in actmapinit:
            self.actmap(v)        
        
        super().__init__(samples, decmap=self.decmap, expmap=self.expmap, \
                actmap=self.actmap)

    def all_decstates(self):
        """
        Returns a sorted list of all decision states
        """
        return list(map(itemgetter(0), \
                    sorted(self.decmap.state_vals.items(),key=itemgetter(1))))
                    
    def all_expstates(self):
        """
        Returns a sorted list of all expectation states
        """
        return list(map(itemgetter(0), \
                    sorted(self.expmap.state_vals.items(),key=itemgetter(1))))
                    
    def all_actions(self):
        """
        Returns a sorted list of all decision states
        """
        return list(map(itemgetter(0), \
                    sorted(self.actmap.state_vals.items(),key=itemgetter(1))))