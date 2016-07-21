""" 
=============================================
Sample manipulation (:mod:`raam.samples`)
=============================================

This package provides tools for constructing, validating, generating, 
and manipulating samples.
"""
import json
import numpy as np
import random
import io
import abc
import operator
import itertools
import collections
from copy import copy

def _itlen(iterator):
    return sum(1 for _ in iterator)

class FormattingError(Exception):
    """ Represents a formatting error. """
    pass

class Sample:
    """
    A sample that starts in a decision state
    """
    def __init__(self, statefrom, action, stateto, reward, weight, step, run):
        self.statefrom = statefrom
        self.action = action
        self.stateto = stateto
        self.reward = reward
        self.weight = weight
        self.step = step
        self.run = run
        
    def map(self, decmap, actmap):
        return Sample(decmap(self.statefrom), actmap(self.action), \
               decmap(self.stateto), self.reward, self.weight, self.step, self.run)
    
    def __str__(self):
        """ String representation """
        # decide whether to use new lines in the description
        if type(self.statefrom) == int and type(self.action) == int:
            return "From: {0.statefrom}; Action: {0.action}; To: {0.stateto}; Reward: {0.reward}; Weight: {0.weight}".format(self)            
        else:
            return "From: {0.statefrom}\nAction: {0.action}\nTo: {0.stateto}\nReward: {0.reward}\nWeight: {0.weight}".format(self)        
    
    def __repr__(self):
        """ String representation """            
        return self.__str__()

class Samples(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def samples(self):
        """ Returns an iterator over decision samples.  """
        pass
    
    @abc.abstractmethod
    def initialsamples(self):
        """ Returns an iterator over the initial samples.  """
        pass
        
    @abc.abstractmethod
    def add_sample(self, sample):
        """ Adds an transition sample.  """
        pass
        
    @abc.abstractmethod
    def add_initial(self, decstate):
        """ Adds and initial state """
        pass
        
    @abc.abstractmethod
    def purge_samples(self):
        """ Removes all decision samples """
        pass
        
    @abc.abstractmethod
    def purge_initial(self):
        """ Removes all initial state samples """
        pass
        
    @abc.abstractmethod
    def copy(self,samples=True,initial=True):
        """
        Returns a copy of the samples that can be modified
        
        Parameters
        ----------
        samples : bool optional
            Whether to copy transition samples
        initial : bool, optional
            Whether to copy initial states
        """
        pass
    
    def merge(self, samples):
        """ 
        Merges samples from another set into this one.
    
        Parameters
        ----------
        samples : Samples
            The second set of samples
        """
        for d in samples.samples():
            self.add_sample(copy(d))
            
    
    def validate(samples):
        """ 
        Checks the structure of the samples and returns basic statistics 
        in a dictionary.
        
        Raises a formatting error when the data is incorrect.
    
        Returns
        -------
        samples : int
            Number of decision states
        expStates : int
            Number of expectation states
    
        See also
        --------
        create_test_sample
        """
        runs = len(set(d.run for d in samples.samples()))
        
        return {'samples':_itlen(samples.samples()), \
                'runs':runs}

    def statistics(samples, discount):
        """
        Computes statistics of the samples, such as the expected reward 
        and the maximal number of steps. It also computes the average 
        return of a sample set. The discount is applied based
        on the step value of each transition sample.
    
        Parameters
        ----------
        samples : dict
            Results from a simulation. These are assumed to follow the structure
            described and checked in 'validate_samples'
        discount : float
            The discount factor
    
        Returns
        -------
        max_step : int
            Maximal number of steps of each run
        sum_rewards : float
            Sum of rewards of each run
        min_reward : float
            Minimal reward encountered for each run
        max_reward : float
            Maximal reward encountered for each run
        mean_return : float
            Mean return over the number of runs
    
        Notes
        -----
        Does not work for extended samples with more than one transition per state. 
    
        The method returns a dictionary
    
        See Also
        --------
        validate_samples
        """
        sums = {}
    
        k = lambda o: o.run

        # arrange samples by runs to make it easy to compute statistics for each 
        runs = [ list(run) for _,run in \
                itertools.groupby(sorted(samples.samples(),key=k),key=k)]

        mean_return = sum(e.reward * (discount ** e.step) \
                                for e in samples.samples()) / len(runs)
        
        return {'mean_return' : mean_return, \
                'max_step' : [max(e.step for e in run) for run in runs], \
                'sum_rewards' : [sum(e.reward for e in run) for run in runs], \
                'min_reward' : [min(e.reward for e in run) for run in runs], \
                'max_reward' : [max(e.reward for e in run) for run in runs]} 


class MemSamples(Samples):
    """
    Simple representation of samples in memory. Supports encoding and decoding 
    as JSON.
    """
    
    def __init__(self,samples=None,init_samples=None):
        """ 
        Creates empty sample dictionary and returns it.
        
        Can take arguments that describe the content of the samples.
        """
        self.dec_samples = samples if samples is not None else []
        self.init_samples = init_samples if init_samples is not None else []

    def encode_json(samples, validate=True):
        """
        Encodes samples into a json file which can be saved and used by other tools
    
        Parameters
        ----------
        validate : bool, optional
            Validate samples before they are saved
    
        Returns
        -------
        out : string
            JSON encoding of the samples
        """
        
        if validate:
            samples.validate()

        return json.dumps({ 'samples':samples.dec_samples, 
                            'initial':samples.init_samples } )
        
    @staticmethod
    def decode_json(jsonstring, validate=True):
        """
        Decodes samples from a json file to a dictionary
    
        Parameters
        ----------
        jsonstring : string
            JSON encoding of the samples
        validate : bool, optional
            Validate samples after they are loaded
    
        Returns
        -------
        out : MemSamples
            Samples loaded in MemSamples
        """
        samples = MemSamples()
        loaded = json.loads(jsonstring)
        samples.dec_samples = loaded['samples']
        samples.init_samples = loaded['initial']
        if validate:
           samples.validate()
        return samples
        
    def samples(self):
        """ Returns an iterator over transition samples.  """
        return (s for s in self.dec_samples)
        
    def initialsamples(self):
        """ Returns samples of initial decision states.  """
        return (s for s in self.init_samples)

    def add_sample(self, decsample):
        """ Adds a decision sample.  """
        self.dec_samples.append(decsample)
        
    def add_initial(self, decstate):
        """ Adds an initial state.  """
        self.init_samples.append(decstate)
        
    def purge_samples(self):
        """ Removes all decision samples """
        self.dec_samples = []
    
    def purge_initial(self):
        """ Removes all initial state samples """
        self.init_samples = []
        
    def copy(self,samples=True,initial=True):
        """
        Returns a copy of the samples that can be modified
        """
        des = list(self.samples()) if samples else None
        ins = list(self.initialsamples()) if initial else None
        return MemSamples(des,ins)


identity = lambda x: x
class SampleView(Samples):
    """
    Transforms samples to a different view. Can be used to add and retrieve
    samples using a different format.
    
    Parameters
    ----------
    samples : raam.Samples
        Samples to be viewed and possibly added
    statemap : function, optional
        A function that maps decision states to their new representation
    statemapinv : function, optional
        A function that maps the new representation of decision states to 
        the underlying format
    actmap : function, optional
        A function that maps actions to their new representation 
    actmapinv : function, optional
        A function that maps the new representation of actions to 
        the underlying format
    readonly : bool, false
        Whether the view is only readonly and it cannot be modified.

    Notes
    -----
    All mapping functions are initialized to identities by default.
    
    The inverse functions (...inv) need to be specified only when the view is 
    used to add more samples to the representation.
    """
    def __init__(self,samples,\
                    statemap=identity,statemapinv=identity, \
                    actmap=identity,actmapinv=identity, \
                    readonly=False):
        
        self._samples        = samples
        self._statemap       = statemap
        self._statemapinv    = statemapinv
        self._actmap         = actmap
        self._actmapinv      = actmapinv
        self._readonly      = readonly

    def samples(self):
        """
        Returns an iterator over decision samples.
        """
        return (s.map(self._statemap,self._actmap) \
                    for s in self._samples.samples())
        
    def initialsamples(self):
        """
        Returns samples of initial decision states.
        """
        return (self._statemap(s) for s in self._samples.initialsamples())

    def add_sample(self, decsample):
        """
        Adds a decision sample.
        """
        if not self._readonly:
            s = decsample.map(self._statemapinv,self._actmapinv)
            self._samples.add_dec(s)
        else:
            raise NotImplementedError('Cannot modify readonly view.')
        
    def add_initial(self, decstate):
        """
        Adds an initial state.
        """
        if not self._readonly:
            self._samples.add_initial(self._statemapinv(decstate))
        else:
            raise NotImplementedError('Cannot modify readonly view.')                        
        
    def purge_samples(self):
        """
        Removes all decision samples
        """
        raise NotImplementedError('Not supported.')
    
    def purge_initial(self):
        """
        Removes all initial state samples
        """
        raise NotImplementedError('Not supported.')
        
    def copy(self,dec=True,initial=True):
        """
        Returns a copy of the samples that can be modified
        """
        raise NotImplementedError('Not supported.')


