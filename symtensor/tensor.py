#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
'''
Symtensor tensor object
'''

import copy
import sys
import numpy as np
import time
from symtensor.symlib import *
from symtensor.tools import utills, logger
from symtensor.tools.path import einsum_path
from symtensor.internal import *

import tensorbackends as backends

class tensor:
    """
    Symmetric tensor class. Stores group symmetric tensors in a reduced represented, along with symmetry information.

    Attributes
    ----------
    array: tensorbackends.interface.Tensor
        Tensor data, represented as numpy array or CuPy/Cyclops object, wrapped as a tensorbackends Tensor. The tensor is order nP+nS-1 where nP is the order of the represented tensor object and nS is the number of modes associated with the symmetry group.

    sym: A 4-tuple (``sign_strings``, ``symmetry_sectors``, ``rhs_value``, ``G``):
                - ``sign_strings`` : A string of "+/-/0" to denote the arithmetic relations between all symmetry sectors. Note character 0 denotes no symmetry for one dimension
                - ``symmetry_sectors`` : A sequence of symmetry sector each symmetric dimension. The symmetry sector for each symmetric dimension can be:
                    - A sequence of :class:`int` for 1D Cn or Zn symmetry.
                    - A sequence of numpy 1D array with same length for higher dimensional product symmetry of Cn or Zn.
                - ``rhs_value`` :  The net symmetry expected in this tensor. Depending on the type in symmetry sector in `symmetry_sector`, it can be:
                    - A integer for 1D Cn or Zn symmetry.
                    - An numpy 1D array for higher dimensional product symmetry of Cn or Zn.
                    - None for symmetry sectors adding up to 0 or 0 vectors.
                - ``G`` : The modulo value for the symmetry conservation relation, optional. Depending on the symmetry sector, it can be:
                    - None for no modulo operation of net symmetry computation.
                    - An integer for 1D Cn/Zn symmetry
                    - A sequence of numpy 1D arrays for higher dimensional product symmetry. The length of the sequence should be equal to the length of each 1D array.
                      Taking (A+B-C-D) mod (G) = rhs as an example. If A/B/C/D/rhs are all integers (1D symmetry), G is expected to be an integer.
                      If A/B/C/D/rhs are all 3D vector (3D product symmetry). G is expected to be 3 3D vectors [G1, G2, G3] and modulus operation
                      refers to there existing integers n1, n2, n3 such that A+B-C-D-n1G1-n2G2-n3G3 = rhs. 

    """
    def __init__(self, array, sym=None, slib=None, backend=tn):
        self.array = array
        self.sym = tuple(sym)
        self._sym = utills._cut_non_sym_sec(sym)
        if sym is None:
            self.ndim = self.array.ndim
            self.shape = self.array.shape
        else:
            self.ndim = self.nsym + self.n0sym
            if self.nsym !=0:
                self.shape = self.array.shape[self.nsym-1:]
            else:
                self.shape = self.array.shape

        if slib is None:
            if backend.name in irrep_map_cache_dict:
                self.irrep_map_cache = irrep_map_cache_dict[backend.name]
            else:
                self.irrep_map_cache = irrep_map_cache(tn)
                irrep_map_cache_dict[backend.name] = self.irrep_map_cache
        else:
            self.irrep_map_cache = slib

    def __getattr__(self, item):
        if hasattr(self.array, item):
            return getattr(self.array, item)
        else:
            raise AttributeError("attribute %s not found"%item)

    @property
    def backend(self):
        return self.array.backend

    @property
    def nsym(self):
        if self._sym is None:
            return 0
        else:
            return len(self._sym[0])

    @property
    def n0sym(self):
        if self.sym is None:
            return 0
        else:
            return self.sym[0].count('0')

    def norm(self):
        '''compute the Frobenius norm of the tensor'''
        return self.backend.norm(self.array)

    def get_irrep_map(self, sym=None):
        if sym is None: sym=self.sym
        return self.irrep_map_cache.get_irrep_map(sym)

    def _as_new_tensor(self, x):
        newtensor = tensor(x, self.sym, backend=self.array.backend, \
                          slib=self.irrep_map_cache)
        return newtensor

    def transpose(self, *axes):
        '''transposing the tensor with specified order'''
        ndim = self.ndim
        assert (len(axes)==ndim), "number of axes does not match the number of symmetry sector"
        if self.sym is None:
            temp = self.array.transpose(axes)
            new_sym = None
        else:
            sign_strings = ''.join([self.sym[0][i] for i in axes])
            sa = DUMMY_STRINGS[:ndim]
            sa_ = utills._cut_non_sym_symbol(sa, self.sym[0])
            sinput = sa_.upper()[:-1] + sa

            sout = ''.join([sa[i] for i in axes])
            sout_ = utills._cut_non_sym_symbol(sout, sign_strings)
            new_symrange = []
            for i in sout_:
                idx = sa_.find(i)
                new_symrange.append(self.sym[1][idx])

            soutput = sout_.upper()[:-1] + sout
            if all([char in sinput for char in soutput]):
                new_order = [sinput.find(char) for char in soutput]
                temp = self.array.transpose(tuple(new_order))
            else:
                irrep_map = self.get_irrep_map()
                sub = sinput + ',' + sa_.upper() + '->' + soutput
                temp = self.backend.einsum(sub, self.array, irrep_map)
            new_sym = [sign_strings, new_symrange, self.sym[2], self.sym[3]]
        return tensor(temp, new_sym, self.irrep_map_cache, backend=backend)

    def ravel(self):
        return self.array.ravel()

    def diagonal(self, preserve_shape=False):
        '''get the diagonal component for tensor with two symmetry sectors, if preserve_shape, will return the matrix with diagonal components'''
        if self.n0sym != 0:
            raise NotImplementedError("diagonal not well defined with non-symmetric sector")
        lib = self.backend
        if self.ndim == self.array.ndim:
            if preserve_shape:
                return tensor(lib.diag(lib.diag(self.array)), self.sym, self.irrep_map_cache, backend=self.array.backend)
            else:
                return lib.diag(self.array)
        elif self.ndim ==2 and self.shape[-1]==self.shape[-2]:
            length = self.shape[-1]
            if preserve_shape:
                p = self.backend.eye(length)
                temp = self.backend.einsum('kij,ij->kij',self.array, p)
                return self._as_new_tensor(temp)
            else:
                temp = self.backend.einsum('kii->ki', self.array)
                return temp
        else:
            raise NotImplementedError("diagonal not defined with more than two symmetry sectors")

    def get_aux_sym_range(self, idx, phase=1):
        '''
        Compute the range for auxillary index

        Parameters:
        -----------
            idx : list
                list of indices which is subset of [0, 1, ..., self.ndim-1]
            phase : int
                +1 or - 1, denoting whether terms appear on left-hand side (+1) or right-hand side (-1)
        '''
        return get_aux_sym_range(self.sym, idx, phase)

    def copy(self):
        return self._as_new_tensor(self.array)

    def conj(self):
        """compute conjugate of the array"""
        return self._as_new_tensor(self.array.conj())

    def __add__(self, x):
        if isinstance(x, tensor):
            return self._as_new_tensor(self.array+x.array)
        else:
            return self._as_new_tensor(self.array+x)

    def __sub__(self, x):
        if isinstance(x, tensor):
            return self._as_new_tensor(self.array-x.array)
        else:
            return self._as_new_tensor(self.array-x)

    def __neg__(self):
        return self._as_new_tensor(-self.array)

    def __mul__(self, x):
        return self._as_new_tensor(self.array*x)

    __rmul__ = __mul__

    def __div__(self, x):
        if isinstance(x, tensor):
            return self._as_new_tensor(self.array/x.array)
        else:
            return self._as_new_tensor(self.array/x)

    __truediv__ = __div__

    def __getitem__(self, key):
        if self.sym is None:
            return self._as_new_tensor(self.array[key])
        new_symbol = ''
        new_sym_range = []
        if self.sym[2] is None:
            new_rhs = 0
        else:
            new_rhs = self.sym[2]
        arr = self.array[key]
        if isinstance(key, (slice, int)):
            key = [key]
        if len(key)>= self.nsym:
            key = key[:self.nsym-1]

        symcount = 0
        nsymcount = 0
        for xs, s in enumerate(self.sym[0]):
            if s=='0':
                new_symbol += '0'
                nsymcount += 1
                continue
            elif symcount>len(key)-1:
                count = symcount + nsymcount
                new_symbol += self.sym[0][count:]
                new_sym_range += self.sym[1][symcount:]
                break
            elif isinstance(key[symcount], (int, np.int)):
                new_symbol += '0'
                new_rhs = new_rhs + self.sym[1][symcount][key[symcount]] * (-1+2*(s=='-'))
                symcount += 1
            else:
                new_symbol += s
                new_sym_range.append(self.sym[1][symcount][key[symcount]])
                symcount += 1

        new_sym = [new_symbol, new_sym_range, new_rhs, self.sym[3]]
        n0sym = new_symbol.count('0')
        nsym = len(new_symbol) - n0sym
        if arr.ndim == 2*nsym-1+n0sym:
            return array(arr, new_sym, backend=arr.backend)
        else:
            return arr

    def __setitem__(self, key, value):
        self.array[key] = value

    def put(self, idx, val):
        self.backend.put(self.array, idx, val)

    def make_dense(self):
        if self.sym is None: return self.array
        irrep_map  = self.irrep_map_cache.get_irrep_map(self.sym)
        ndim  = self.ndim
        s_A = DUMMY_STRINGS[:ndim]
        s_A_ = utills._cut_non_sym_symbol(s_A, self.sym[0])
        Ain = s_A_[:-1].upper() + s_A
        Aout = s_A_.upper() + s_A
        sub = Ain + ',' + s_A_.upper() + '->' + Aout
        array = self.backend.einsum(sub, self.array, irrep_map)
        return array

    def enforce_sym(self):
        if self.sym is None: return self
        irrep_map  = self.irrep_map_cache.get_irrep_map(self.sym)
        ndim  = self.ndim
        s_A = DUMMY_STRINGS[:ndim]
        s_A_ = utills._cut_non_sym_symbol(s_A, self.sym[0])
        Ain = s_A_.upper() + s_A
        Aout = s_A_[:-1].upper() + s_A
        sub = Ain + ',' + s_A_.upper() + '->' + Aout
        sparse = self.make_dense()
        self.array = self.backend.einsum(sub, sparse, irrep_map)
        sparse = None
