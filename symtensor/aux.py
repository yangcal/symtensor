#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
'''
Symtensor auxiliary interface functions mimiciing numpy functionality, mostly tensor constructors.
'''


import copy
import sys
import numpy as np
import time
from symtensor.symlib import *
from symtensor.tools import utills, logger
from symtensor.tools.path import einsum_path
from symtensor.internal import _einsum
from symtensor.internal import *
from symtensor.tensor import tensor

import tensorbackends as backends


def array(arr, sym=None, backend=None):
    """
    Create a copy of a symmetric tensor based on tensor data and symmetry

    Parameters
    ----------
    arr: tensor_like
        Input tensor like object

    sym: 4-tuple
        Specification of symmetry, refer to main tensor constructor for details

    Returns
    -------
    output: tensor
        A tensor object with specified symmetry.
    """
    return tensor(arr, sym, backend)


def get_full_shape(shape, sym=None):
    """
    Return shape of reduced form of tensor given shape of a symmetry block and the symmetry specification

    Parameters
    ----------
    shape: list(int)
        Shape of symmetric block

    sym: 4-tuple
        Specification of symmetry, refer to main tensor constructor for details

    Returns
    -------
    output: list(int)
        Shape of the reduced form, with symmetry sectors stored first

    Examples
    --------
    >>> import symtensor as st
    >>> st.get_full_shape([3,4,5],['++-',[2,2,2],0,2])
    (2,2,3,4,5)
    """
    if sym is not None:  assert(len(sym[0])==len(shape))
    if sym is None:
        return shape
    else:
        full_shape = [len(i) for i in sym[1][:-1]] + list(shape)
        return tuple(full_shape)

def zeros(shape, sym=None, dtype=float, backend=None):
    """
    Create a zero symmetric tensor with specified symmetry

    Parameters
    ----------
    shape: list(int)
        Shape of each symmetric tensor block

    sym: 4-tuple
        Specification of symmetry, refer to main tensor constructor for details

    dtype: type
        Specification of tensor element type

    backend: tensorbackends.backend
        Tensor array backend to use (numpy by default, can be CTF or CuPy)

    Returns
    -------
    output: tensor
        A zeroed out tensor object with specified symmetry.
    """
    full_shape = get_full_shape(shape, sym)
    arr = tn.zeros(full_shape, dtype)
    return array(arr, sym, backend)

def zeros_like(a, dtype=None):
    """
    Create a zero symmetric tensor with same size and symmetry as given tensor

    Parameters
    ----------
    a: symtensor.tensor
        Symmetric tensor to copy

    dtype: type
        Specification of tensor element type

    Returns
    -------
    output: tensor
        A zeroed out tensor object with same properties as a
    """
    if dtype is None: dtype=a.dtype
    return zeros(a.shape, a.sym, dtype, a.array.backend)

def random(shape, sym=None, backend=tn):
    """
    Create a random symmetric tensor with specified symmetry

    Parameters
    ----------
    shape: list(int)
        Shape of each symmetric tensor block

    sym: 4-tuple
        Specification of symmetry, refer to main tensor constructor for details

    backend: tensorbackends.backend
        Tensor array backend to use (numpy by default, can be CTF or CuPy)

    Returns
    -------
    output: tensor
        A random tensor object with specified symmetry.
    """
    full_shape = get_full_shape(shape, sym)
    arr = backend.random.random(full_shape)
    tensor = array(arr, sym)
    tsr_copy = array(arr.copy(), sym)
    tensor.enforce_sym()
    return tensor

def diag(array, sym=None, backend=tn):
    """
    Extract a diagonal part of matrix or create diagonal matrix from vector.
    Intended to act as numpy.diag.
    FIXME/WARNING: functionality not present for some cases/backends

    Parameters
    ----------
    array: tensor-like
        Tensor object, either backend tensor of data or symtensor

    sym: 4-tuple
        Specification of symmetry, refer to main tensor constructor for details

    backend: tensorbackends.backend
        Tensor array backend to use (numpy by default, can be CTF or CuPy)

    Returns
    -------
    output: tensor
        Symmetric tensor
    """
    IS_SYMTENSOR = isinstance(array,tensor)
    if IS_SYMTENSOR:
        nsym = array.nsym
        n0sym = array.n0sym
    else:
        if sym is None:
            nsym = n0sym = 0
        else:
            nsym = len(sym[0]) - sym[0].count('0')
            n0sym = sym[0].count('0')

    if nsym not in [0,1,2]:
        raise ValueError("tensor must be 1d or 2d for diag function call")

    if n0sym !=0:
        raise ValueError("diag not well defined for tensor with non-symmetry sector")

    if IS_SYMTENSOR:
        return array.diagonal()
    else:
        if nsym == 0 or nsym==1:
            return backend.diag(array)
        else:
            if array.ndim==3:
                assert(array.shape[1]==array.shape[2])
                out = backend.einsum('kii->ki', array)
                return out
            elif array.ndim==2:
                if not all(len(i)==array.shape[0] for i in sym[1]):
                    raise ValueError("The first dimension of the array must be equal to the number of symmetry sectors")
                raise ValueError("Cannot perform diag with specified parameters")
                #FIXME: code below does not make sense, _sym needs to be defined
                #out = backend.einsum('ki,ij->kij', array, backend.eye(array.shape[-1]))
                #return tensor(out, _sym, backend=backend)
            else:
                raise ValueError("Symmetry not compatible with input array")


def einsum(subscripts, *operands):
    """
    Contract symtensor objects according to index strings.
    Functions as numpy.einsum.

    Parameters
    ----------
    subscripts: string
        Description of contraction indices, with symmetry handled implicitly.

    operands: list of symtensor.tensor objects
        Symmetric tensors to contract

    Returns
    -------
    output: tensor
        Symmetric tensor
    """

    newsub = copy.copy(subscripts).replace(' ', '')
    if '->' not in newsub:
        newsub += '->'
    sublist= newsub.replace('->',',').split(',')
    if len(sublist) != len(operands)+1:
        raise ValueError("subscript inputs not matching number of operands")

    use_symmetry = 0
    use_non_symmetry = 0
    for ki, i in enumerate(operands):
        if isinstance(i,tensor):
            if i.array.ndim == len(sublist[ki]):
                use_non_symmetry += 1
            if i.ndim == len(sublist[ki]):
                use_symmetry += 1
        else:
            use_non_symmetry += 1
    if use_symmetry!= len(operands) and use_non_symmetry != len(operands):
        raise TypeError("mixed symmetric and non-symmetric labels found in the subscript, \
                        please switch to fully non-symmetric or symmetric notation")
    if use_non_symmetry==len(operands):
        backend = infer_backend(operands[0])
        tmp_operands = [getattr(i, 'array', i) for i in operands]
        return backend.einsum(subscripts, *tmp_operands)
   
    contraction_list = einsum_path(subscripts, *operands)
    operands = [v for v in operands]
    for inds, idx_rm, einsum_str in contraction_list:
        if len(inds) > 2:
            raise ValueError("einsum_path not able to return pairwise contraction, \
                              Please manually perform pairwise contraction")
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str= contraction
        tmp_operands = [operands.pop(x) for x in inds]
        new_view = _einsum(einsum_str, *tmp_operands)
        operands.append(new_view)
        del tmp_operands, new_view
    return operands[0]


def static_partition(tasks,backend):
    assert(backend.name == 'ctf')
    comm = backend.comm()
    rank = comm.rank()
    size = comm.np()
    segsize = (len(tasks)+size-1) // size
    start = rank * segsize
    stop = min(len(tasks), start+segsize)
    ntasks = min(len(tasks),segsize)
    return tasks[start:stop], ntasks

def fromfunction(func, shape, sym=None, dtype=float, backend=tn, **kwargs):
    out = zeros(shape, sym, dtype=dtype, backend=backend)
    nsym = out.nsym
    if backend.name == 'ctf':
        if nsym==0:
            itask, ntasks = static_partition(np.arange(out.size),backend)
            index = np.unravel_index(itask, shape)
            trunk_data = func(*index, **kwargs)
            out.put(itask, trunk_data.ravel())
        else:
            sym_shape = list(out.array.shape[:nsym-1])
            tasks, ntasks = static_partition(np.arange(np.prod(sym_shape)),backend)
            trunk_size = np.prod(shape)
            for i in range(ntasks):
                if i >= len(tasks):
                    out.put([], [])
                    continue
                itask = tasks[i]
                idx = np.unravel_index(itask, sym_shape)
                trunk_data = func(*idx, **kwargs)
                trunk_idx = itask * trunk_size + np.arange(trunk_size)
                out.put(trunk_idx, trunk_data.ravel())
    else:
        if out.nsym==0:
            out.array = np.fromfunction(func, shape, **kwargs)
        else:
            kwargs.pop('dtype', None)
            sym_shape = list(out.array.shape[:nsym-1])
            ntasks = np.prod(sym_shape)
            trunk_size = np.prod(shape)
            for i in range(ntasks):
                idx = np.unravel_index(i, sym_shape)
                trunk_data = func(*idx, **kwargs)
                trunk_idx = i * trunk_size + np.arange(trunk_size)
                out.put(trunk_idx, trunk_data.ravel())
        
    return out

def frombatchfunc(func, shape, all_tasks, nout=1, out=None, sym=None, dtype=float, backend=tn, **kwargs):
    if isinstance(shape[0], list) or isinstance(shape[0], tuple):
        shape_list = shape
        nout = len(shape)
    else:
        shape_list = [shape,] * nout

    if sym is None:
        sym_list = [sym,] *nout
    elif isinstance(sym[0], str):
        sym_list = [sym,]*nout
    else:
        sym_list = sym

    if out is None:
        if nout==1:
            out = zeros(shape, sym, dtype=dtype, backend=backend)
        else:
            out = [zeros(shape_list[i], sym_list[i], dtype=dtype, backend=backend) for i in range(nout)]
    else:
        if isinstance(out, (list, tuple)):
            nout = len(out)
        else:
            nout = 1

    if backend.name == 'ctf':
        tasks, ntasks = static_partition(all_tasks,backend)
        for itask in range(ntasks):
            if itask>= len(tasks):
                if nout ==1:
                    out.put([], [])
                else:
                    for i in range(nout):
                        out[i].put([], [])
                continue
            task = tasks[itask]
            if isinstance(task, (tuple, list)):
                inds, vals = func(*task, **kwargs)
            else:
                inds, vals = func(task, **kwargs)

            if nout ==1:
                out.put(inds.ravel(), vals.ravel())
            else:
                for i in range(nout):
                    out[i].put(inds[i].ravel(), vals[i].ravel())
    else:
        for itask in all_tasks:
            if isinstance(itask, (tuple, list)):
                inds, vals = func(*itask, **kwargs)
            else:
                inds, vals = func(itask, **kwargs)
                
            if nout ==1:
                out.put(inds.ravel(), vals.ravel())
            else:
                for i in range(nout):
                    out[i].put(inds[i].ravel(), vals[i].ravel())
            inds = vals = None
    return out
