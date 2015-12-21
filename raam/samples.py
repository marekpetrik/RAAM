""" 
=============================================
Sample manipulation (:mod:`raam.samples`)
=============================================

This package provides tools for constructing, validating, generating, and manipulating samples.
"""
import json
import numpy as np
import random
import io
import abc
import operator
import itertools
import collections

def _itlen(iterator):
    return sum(1 for _ in iterator)

class FormattingError(Exception):
    """ Represents a formatting error. """
    pass

class ExpSample:
    """
    A sample that starts in an expectation state
    """
    def __init__(self,expStateFrom, decStateTo, reward, weight, step, run, other = None):
        self.expStateFrom = expStateFrom
        self.decStateTo = decStateTo
        self.reward = reward
        self.weight = weight
        self.step = step 
        self.run = run
        self.other = other
        
    def map(self, decmap, expmap, actmap):
        return ExpSample(expmap(self.expStateFrom), decmap(self.decStateTo),\
                    self.reward, self.weight, self.step, self.run, self.other)
        
class DecSample:
    """
    A sample that starts in a decision state
    """
    def __init__(self, decStateFrom, action, expStateTo, step, run, other = None):
        self.decStateFrom = decStateFrom
        self.action = action
        self.expStateTo = expStateTo
        self.step = step
        self.run = run
        self.other = other
        
    def map(self, decmap, expmap, actmap):
        return DecSample(decmap(self.decStateFrom), actmap(self.action), \
                expmap(self.expStateTo), self.step, self.run, self.other)
        

class Samples(metaclass=abc.ABCMeta):
    """
    General representation of the samples class
    """
    @abc.abstractmethod
    def expsamples(self):
        """
        Returns an iterator over expectation samples.
        """
        pass
    
    @abc.abstractmethod
    def decsamples(self):
        """
        Returns an iterator over decision samples.
        """
        pass
    
    @abc.abstractmethod
    def initialsamples(self):
        """
        Returns an iterator over the initial samples.
        """
        pass
        
    @abc.abstractmethod
    def add_exp(self, expsample):
        """
        Adds an expectation sample.
        """
        pass
        
    @abc.abstractmethod
    def add_dec(self, decsample):
        """
        Adds a decision sample.
        """
        pass
        
    @abc.abstractmethod
    def add_initial(self, decstate):
        """
        Adds and initial state
        """
        pass
        
    @abc.abstractmethod
    def purge_exp(self):
        """
        Removes all expectation samples
        """
        pass
    
    @abc.abstractmethod
    def purge_dec(self):
        """
        Removes all decision samples
        """
        pass
        
    @abc.abstractmethod
    def purge_initial(self):
        """
        Removes all initial state samples
        """
        pass
        
    @abc.abstractmethod
    def copy(self,exp=True,dec=True,initial=True):
        """
        Returns a copy of the samples that can be modified
        
        Parameters
        ----------
        exp : bool, optional 
            Whether to copy expectation samples
        dec : bool, optional
            Whether to copy decision samples
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
        for d in samples.decsamples():
            self.add_dec(d)
            
        for e in samples.expsamples():
            self.add_exp(e)
            
    
    def validate(samples):
        """ 
        Checks the structure of the samples and returns basic statistics in a dictionary.
        Raises a formatting error when the data is incorrect.
    
        Returns
        -------
        decStates : int
            Number of decision states
        expStates : int
            Number of expectation states
    
        See also
        --------
        create_test_sample
        """
    
        runs = len(set(e.run for e in samples.expsamples()) | set(d.run for d in samples.decsamples()))
        
        return {'decStates':_itlen(samples.decsamples()), \
                'expStates':_itlen(samples.expsamples()), \
                'runs':runs}

    def statistics(samples, discount):
        """
        Computes statistics of the samples, such as the expected reward and the maximal number of steps.
        It also computes the average return of a sample set. The discount is applied based
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
        runs = [ list(run) for _,run in \
                itertools.groupby(sorted(samples.expsamples(),key=k),key=k)]
        mean_return = sum(e.reward * (discount ** e.step) for e in samples.expsamples()) / len(runs)
        
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
    
    def __init__(self,dec_samples=None,exp_samples=None,init_samples=None):
        """ 
        Creates empty sample dictionary and returns it.
        
        Can take arguments that describe the content of the samples.
        """
        self.dec_samples = dec_samples if dec_samples is not None else []
        self.exp_samples = exp_samples if exp_samples is not None else []
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
        return json.dumps({'expSamples':samples.exp_samples, 'decSamples':samples.dec_samples, 'initSamples':samples.init_samples } )
        
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
        samples.dec_samples = loaded['decSamples']
        samples.exp_samples = loaded['expSamples']
        samples.init_samples = loaded['initSamples']
        if validate:
           samples.validate()
        return samples
        
    @staticmethod
    def from_dict(loaded, validate=True):
        """
        Decodes samples from a json file to a dictionary
    
        Parameters
        ----------
        loaded : dictionary
            simple dictionary encoding of the samples
        validate : bool, optional
            Validate samples after they are loaded
    
        Returns
        -------
        out : MemSamples
            Samples loaded in MemSamples
        """
        samples = MemSamples()
        samples.dec_samples = loaded['decSamples']
        samples.exp_samples = loaded['expSamples']
        samples.init_samples = loaded['initSamples']
        if validate:
           samples.validate()
        return samples        
        
    def expsamples(self):
        """
        Returns an iterator over expectation samples.
        """
        return (s for s in self.exp_samples)
    
    def decsamples(self):
        """
        Returns an iterator over decision samples.
        """
        return (s for s in self.dec_samples)
        
    def initialsamples(self):
        """
        Returns samples of initial decision states.
        """
        return (s for s in self.init_samples)

    def add_exp(self, expsample):
        """
        Adds an expectation sample.
        """
        self.exp_samples.append(expsample)
        
    def add_dec(self, decsample):
        """
        Adds a decision sample.
        """
        self.dec_samples.append(decsample)
        
    def add_initial(self, decstate):
        """
        Adds an initial state.
        """
        self.init_samples.append(decstate)
        
    def purge_exp(self):
        """
        Removes all expectation samples
        """
        self.exp_samples = []

    def purge_dec(self):
        """
        Removes all decision samples
        """
        self.dec_samples = []
    
    def purge_initial(self):
        """
        Removes all initial state samples
        """
        self.init_samples = []
        
    def copy(self,dec=True,exp=True,initial=True):
        """
        Returns a copy of the samples that can be modified
        """
        des = list(self.decsamples()) if dec else None
        exs = list(self.expsamples()) if exp else None
        ins = list(self.initialsamples()) if initial else None
        return MemSamples(des,exs,ins)


identity = lambda x: x
class SampleView(Samples):
    """
    Transforms samples to a different view. Can be used to add and retrieve
    samples using a different format.
    
    Parameters
    ----------
    samples : raam.Samples
        Samples to be viewed and possibly added
    decmap : function, optional
        A function that maps decision states to their new representation
    decmapinv : function, optional
        A function that maps the new representation of decision states to 
        the underlying format
    expmap : function, optional
        A function that maps expectation states to their new representation
    expmapinv : function, optional
        A function that maps the new representation of expectation states to 
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
                    decmap=identity,decmapinv=identity, \
                    expmap=identity,expmapinv=identity, \
                    actmap=identity,actmapinv=identity, \
                    readonly=False):
        
        self.samples    = samples
        self.decmap     = decmap
        self.decmapinv  = decmapinv
        self.expmap     = expmap
        self.expmapinv  = expmapinv
        self.actmap     = actmap
        self.actmapinv  = actmapinv
        self._readonly  = readonly

    def expsamples(self):
        """
        Returns an iterator over expectation samples.
        """
        return (s.map(self.decmap,self.expmap,self.actmap) \
                    for s in self.samples.expsamples())
    
    def decsamples(self):
        """
        Returns an iterator over decision samples.
        """
        return (s.map(self.decmap,self.expmap,self.actmap) \
                    for s in self.samples.decsamples())
        
    def initialsamples(self):
        """
        Returns samples of initial decision states.
        """
        return (self.decmap(s) for s in self.samples.initialsamples())

    def add_exp(self, expsample):
        """
        Adds an expectation sample.
        """
        if not self._readonly:
            s = expsample.map(self.decmapinv,self.expmapinv,self.actmapinv)
            self.samples.add_exp(s)
        else:
            raise NotImplementedError('Cannot modify readonly view.')
        
    def add_dec(self, decsample):
        """
        Adds a decision sample.
        """
        if not self._readonly:
            s = decsample.map(self.decmapinv,self.expmapinv,self.actmapinv)
            self.samples.add_dec(s)
        else:
            raise NotImplementedError('Cannot modify readonly view.')
        
    def add_initial(self, decstate):
        """
        Adds an initial state.
        """
        if not self._readonly:
            self.samples.add_initial( \
                        self.decmapinv(decstate))
        else:
            raise NotImplementedError('Cannot modify readonly view.')                        
        
    def purge_exp(self):
        """
        Removes all expectation samples
        """
        raise NotImplementedError('Not supported.')

    def purge_dec(self):
        """
        Removes all decision samples
        """
        raise NotImplementedError('Not supported.')
    
    def purge_initial(self):
        """
        Removes all initial state samples
        """
        raise NotImplementedError('Not supported.')
        
    def copy(self,dec=True,exp=True,initial=True):
        """
        Returns a copy of the samples that can be modified
        """
        raise NotImplementedError('Not supported.')


def create_test_sample():
    """ 
    Creates a test in-memory sample and returns it 
    
    See Also
    --------
    raam.samples.MemSamples
    """
    
    samples = MemSamples()
    
    s_decstates = [[0,1],[1,2],[2,3]]
    s_expstates = [[5,6],[6,7],[7,8]]
    profits = [1,2,3]
    
    for i,(d,e,p) in enumerate(zip(s_decstates, s_expstates, profits)):
        samples.add_exp(ExpSample(e, d, p, 1.0, i, 0))
        samples.add_dec(DecSample(d, 1, e, i, 0))
    return samples

