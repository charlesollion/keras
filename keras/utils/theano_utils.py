from __future__ import absolute_import
import numpy as np
import theano
import theano.tensor as T

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

def shared_zeros(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)

def shared_scalar(val=0., dtype=theano.config.floatX, name=None):
    return theano.shared(np.cast[dtype](val))

def shared_ones(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.ones(shape), dtype=dtype, name=name)

def alloc_zeros_matrix(*dims):
    return T.alloc(np.cast[theano.config.floatX](0.), *dims)

def T_one_hot(t, r=None):
    """
    given a tensor t of dimension d with integer values from range(r), return a
    new tensor of dimension d + 1 with values 0/1, where the last dimension
    gives a one-hot representation of the values in t.
    if r is not given, r is set to max(t) + 1
    * Taken from benanne
    """
    if r is None:
        r = T.max(t) + 1
    ranges = T.shape_padleft(T.arange(r), t.ndim)
    return T.eq(ranges, T.shape_padright(t, 1)) 
