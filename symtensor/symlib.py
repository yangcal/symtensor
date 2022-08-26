#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
'''
Routines for generating delta tensors (irrep maps) and an object for caching them
'''

from symtensor.tools import utills
from symtensor.misc import DUMMY_STRINGS
import itertools
import numpy as np
import copy
#tolerance for equivalence checks
SYM_TOL=1e-6
#map from symmetry signs to ints
sign = {'+':1,'-':-1}

"""
Get irrep map object for a symmetry

Parameters
----------
sym: [string, list(int), int, int]
    Symmetry to represent as irrep map, refer to main tensor constructor for specification

Returns
----------
tensor representing symmetry
"""
def sym_to_irrep_map(sym, backend):
    rank = getattr(backend, "rank", 0)
    sign_string, sym_range, rhs, modulus = utills._cut_non_sym_sec(sym)
    shape = [len(i) for i in sym_range]
    delta = backend.zeros(shape)
    if isinstance(sym_range[0][0], (int, np.integer)):
        ndim = 1
    else:
        ndim = len(sym_range[0][0])

    if rank==0:
        def inv(modulus):
            vol = np.linalg.det(modulus)
            modulus_inv = np.asarray([np.cross(modulus[i-2],modulus[i-1])/vol for i in range(ndim)]).T
            return modulus_inv
        val = 0
        nsec =  len(sym_range)
        for i in range(nsec):
            shape = [1,] * nsec + [ndim]
            shape[i] = len(sym_range[i])
            temp = np.asarray(sym_range[i]).reshape(shape) * sign[sign_string[i]]
            val = val + temp

        if rhs is not None:
            val -= rhs

        val = val.reshape(-1, ndim)
        if modulus is not None:
            if ndim==1:
                val = np.mod(val, modulus)
            else:
                modulus_inv = inv(modulus)
                val = np.dot(val, modulus_inv)
                val = val - np.rint(val)

        val = np.sum(abs(val), axis=1)
        idx = np.where(val.ravel()<SYM_TOL)[0]
        fill = np.ones(len(idx))
        shape = [len(i) for i in sym_range]

        backend.put(delta, idx, fill)
    else:
        backend.put(delta, [], [])
    return delta


def fold_sym_range(sym_range, modulus):
    """Return the Minimal represetation of a symmetry range"""
    if modulus is None:
        if isinstance(sym_range[0][0], int):
            return np.unique(sym_range)
        else:
            val = np.unique(np.round_(sym_range, 10),axis=0)
            return val
    else:
        if isinstance(modulus, int):
            return np.unique(np.mod(sym_range, modulus))
        else:
            vol = np.linalg.det(modulus)
            ndim = len(modulus)
            pbc_inv = np.asarray([np.cross(modulus[i-2], modulus[i-1])/vol for i in range(ndim)]).T
            sym_array = np.asarray(sym_range)
            val = np.dot(sym_array, pbc_inv)
            val = np.round_(val, 10)
            val = val - np.floor(val)
            val = np.round_(val, 10)
            val = np.unique(val, axis=0)
            return np.dot(val, modulus)

def get_aux_sym_range(sym, idx, phase=1):
    """compute the range for the auxillary index based on two sides of algebraic equations for the given indices"""
    sign_string, sym_range, rhs, modulus = utills._cut_non_sym_sec(sym)

    sA = DUMMY_STRINGS[:len(sym[0])]
    sA_ = utills._cut_non_sym_symbol(sA, sym[0])
    new_idx =  []
    for i in idx:
        if sym[0][i] == '0': continue
        new_idx.append(sA_.find(sA[i]))
    left_idx, right_idx = new_idx, [i for i in range(len(sym_range)) if i not in new_idx]
    nleft, nright = len(left_idx), len(right_idx)
    out_left, out_right = [0,]*2
    if isinstance(sym_range[0][0], (int, np.integer)):
        ndim = 1
    else:
        ndim = len(sym_range[0][0])

    for ki, i in enumerate(left_idx):
        shape = [1,] * nleft + [ndim]
        shape[ki] = len(sym_range[i])
        out_left = out_left + np.asarray(sym_range[i]).reshape(shape) * sign[sign_string[i]] * phase

    for kj, j in enumerate(right_idx):
        shape = [1,] * nright + [ndim]
        shape[kj] = len(sym_range[j])
        out_right = out_right +  np.asarray(sym_range[j]).reshape(shape) * sign[sign_string[j]] * -1 * phase# flip the sign

    out_left, out_right = out_left.reshape(-1,ndim), out_right.reshape(-1,ndim)
    if rhs is not None: out_right += rhs
    out_left = fold_sym_range(out_left, modulus)
    out_right = fold_sym_range(out_right, modulus)
    if ndim ==1:
        out_left = out_left.reshape(-1)
        out_right = out_right.reshape(-1)
    aux_range = merge_sym_range(out_left, out_right)
    return aux_range

def merge_sym_range(range_A, range_B, modulus1=None, modulus2=None):
    '''In cases when only one tensor has moduli, symmetry ranges needs to be folded'''
    if modulus1 is None and modulus2 is None:
        return _merge_sym_range(range_A, range_B)
    elif modulus1 is not None and modulus2 is not None:
        assert(np.allclose(modulus1, modulus2))
        return _merge_sym_range(range_A, range_B)
    elif modulus1 is None and modulus2 is not None:
        rA = fold_sym_range(range_A, modulus2)
        return _merge_sym_range(range_B, rA)
    else:
        rB = fold_sym_range(range_B, modulus1)
        return _merge_sym_range(range_A, rB)

def _merge_sym_range(range_A, range_B):
    delta = abs(np.asarray(range_A)[:,None]-np.asarray(range_B)[None:])
    if delta.ndim !=2:
        delta = np.sum(delta, axis=2)
    idx = np.where(delta<SYM_TOL)
    merged_range = [range_A[i] for i in idx[0]]
    return merged_range

def check_sym_equal(sym1, sym2):
    sign_string1, sym_range1, rhs1, modulus1 = utills._cut_non_sym_sec(sym1)
    sign_string2, sym_range2, rhs2, modulus2 = utills._cut_non_sym_sec(sym2)
    if rhs1 is None: rhs1 = 0
    if rhs2 is None: rhs2 = 0

    flip = {'+':'-', '-':'+'}

    if modulus1 is not None and modulus2 is not None:
        mod_delta = np.asarray(modulus1) - np.asarray(modulus2)
        EQUAL = np.linalg.norm(mod_delta) < SYM_TOL
    elif modulus1 is None and modulus2 is None:
        EQUAL = True
    else:
        EQUAL = False
    MATCH = np.linalg.norm(rhs1-rhs2)<SYM_TOL
    FLIP_MATCH = np.linalg.norm(rhs1+rhs2)<SYM_TOL
    sign2_flip = ''.join([flip[i] for i in sign_string2])
    EQUAL = EQUAL and (MATCH or FLIP_MATCH) and len(sign_string1)==len(sign_string2)
    if not EQUAL:
        return (EQUAL, None)
    EQUAL = False
    for idx, order in enumerate(itertools.permutations(range(len(sign_string1)))):
        new_strings = ''.join([sign_string1[i] for i in order])
        if (new_strings == sign_string2 and MATCH) or (new_strings == sign2_flip and FLIP_MATCH):
            lst = []
            for ki, i in enumerate(order):
                try:
                    sym_delta = np.asarray(sym_range1[i]) - np.asarray(sym_range2[ki])
                    lst.append(np.linalg.norm(sym_delta)<SYM_TOL)
                except:
                    lst.append(False)
            EQUAL = all(lst)
            if EQUAL: break

    if EQUAL and idx!=0:
        return (EQUAL, order)
    else:
        return (EQUAL, None)

def fuse_symbackend(symbackend1, symbackend2):
    if symbackend1 is None:
        return symbackend2
    elif symbackend2 is None:
        return symbackend1
    fused_backend = symbackend1.copy()
    for ki, i in enumerate(symbackend2.sym_lst):
        SYM_INCLUDED = [check_sym_equal(symi, i)[0] for symi in symbackend1.sym_lst]
        if not any(SYM_INCLUDED):
            fused_backend.sym_lst.append(i)
            fused_backend.irrep_map_lst.append(symbackend2.irrep_map_lst[ki])
    return fused_backend

def make_irrep_map_lst(symbackend, sym1, sym2, sym_string_lst):
    sym1_ = utills._cut_non_sym_sec(sym1)
    sym2_ = utills._cut_non_sym_sec(sym2)
    sign_string1, sym_range1, rhs1, modulus1 = sym1_
    sign_string2, sym_range2, rhs2, modulus2 = sym2_
    s_A, s_B, s_C = sym_string_lst

    contracted = sorted(set(s_A) & set(s_B))
    res_A = set(s_A) - set(contracted)
    res_B = set(s_B) - set(contracted)
    delta_strings = [s_A, s_B]
    delta_tensor = [symbackend.get_irrep_map(sym1_), symbackend.get_irrep_map(sym2_)]
    if len(contracted) > 1 and len(res_A)>1 and len(res_B)>1:
        # when more than two symmetry sectors are contracted out and the delta does not exist, auxillary index iss generated
        s_q = ''.join(contracted) + 'Q'
        delta_strings.append(s_q)
        idxa = [s_A.find(i) for i in contracted]
        idxb = [s_B.find(i) for i in contracted]
        if sym1_[0][idxa[0]] != sym2_[0][idxb[0]]:
            phase = -1
        else:
            phase = 1
        auxa = get_aux_sym_range(sym1_, idxa)
        auxb = get_aux_sym_range(sym2_, idxb, phase)
        aux_range = merge_sym_range(auxa, auxb, modulus1, modulus2)
        # when auxillary range is different from the two tensors, pick the shared one for most compact representation
        sign_string = ''.join([sym1_[0][i] for i in idxa]) + '-'
        sym_range = [sym1[1][i] for i in idxa]
        sym_range.append(aux_range)
        if sym1[3] is not None:
            symq = sym1[3]
        else:
            symq = sym2[3]
        aux_sym = [sign_string, sym_range, None, symq]
        delta_tensor.append(symbackend.get_irrep_map(aux_sym))
    delta_lst = fuse_delta([delta_strings, delta_tensor], symbackend.backend)
    return delta_lst

def fuse_delta(delta_lst, backend):
    '''generate all delta tensors/strings in sorted order (by length) from the input delta tensors/strings
    eg, ijab, abuv, ijq ---> ijq, abq, uvq, ijab, abuv, ijuv'''
    strings, deltas  = delta_lst
    ndelta = len(deltas)
    string_set = [set(i) for i in strings]
    for iter in range(2):
        for i, j in itertools.combinations(range(ndelta), 2):
            s_i, s_j = strings[i], strings[j]
            ovlp = set(s_i) & set(s_j)
            if len(ovlp) ==0: continue
            s_fused = utills.set_to_str((set(s_i)|set(s_j))-ovlp)
            if len(s_fused)==0 or set(s_fused) in string_set:
                continue
            sub = utills.lst_to_sub([s_i,s_j,s_fused])
            delta = _fuse_delta(sub, deltas[i], deltas[j], backend)
            strings.append(s_fused)
            string_set.append(set(s_fused))
            deltas.append(delta)
        ndelta = len(deltas)
    new_strings = sorted(strings,key=len)
    new_deltas = [deltas[strings.index(i)] for i in new_strings]
    return (new_strings, new_deltas)

def _fuse_delta(sub, delta_a, delta_b, backend):
    """Generate a new delta tensor by tensor contraction between two input delta tensors a and b"""
    temp = backend.einsum(sub, delta_a, delta_b)
    idx = backend.nonzero(temp.ravel())
    delta = backend.zeros(temp.shape)
    fill = np.ones(len(idx))
    rank = getattr(backend, "rank", 0)
    if rank==0:
        backend.put(delta, idx, fill)
    else:
        backend.put(delta, [], [])
    return delta


class irrep_map_cache:
    """
    Class for generating "irrep map" tensors, which represent a particular symmetry object. Caches these objects so they need not be reconstructed repeatedly during iterative computation.

    Attributes
    ----------
    backend: tensorbackends object
        Wrapper for numpy/CuPy/Cyclops to represent irrep map tensors in this cache

    sym_lst: list of [string, list(int), int, int]
        Symmetries to represent as irrep map, refer to main tensor constructor for specification of each symmetry

    irrep_map_list: list of tensors
        Tensors representing irrep maps corresponding to each symmetry in sym_lst, represented with the given backend object
    """

    """
    Constructor

    Parameters
    ----------
    backend: tensorbackends object
        Wrapper for numpy/CuPy/Cyclops to represent irrep map tensors in this cache
    """
    def __init__(self, backend):
        self.backend = backend
        self.sym_lst = []
        self.irrep_map_lst = []

    """
    Adds symmetries to cache / constructs irrep

    Parameters
    ----------
    *sym: list of [string, list(int), int, int]
        Symmetries to represent as irrep map, refer to main tensor constructor for specification of each symmetry
    """
    def update(self, *sym):
        for i in sym:
            self._update(i)
        return self

    """
    Add symmetry to cache / constructs irrep

    Parameters
    ----------
    sym: [string, list(int), int, int]
        Symmetry to represent as irrep map, refer to main tensor constructor for specification
    """
    def _update(self, sym):
        sym_ = utills._cut_non_sym_sec(sym)
        SYM_INCLUDED = [check_sym_equal(symi, sym_)[0] for symi in self.sym_lst]
        if not any(SYM_INCLUDED):
            self.sym_lst.append(sym_)
            irrep_map = sym_to_irrep_map(sym_, self.backend)
            self.irrep_map_lst.append(irrep_map)

    """
    Merge with another irrep map cache

    Parameters
    ----------
    symbackend2: irrep_map_cache
        Another cache to merge with

    Returns
    ----------
    new irrep_map_cache object containing irrep maps in this cache and the given cache
    """
    def __add__(self, symbackend2):
        return fuse_symbackend(self, symbackend2)

    """
    Get irrep map object for a symmetry

    Parameters
    ----------
    sym: [string, list(int), int, int]
        Symmetry to represent as irrep map, refer to main tensor constructor for specification

    Returns
    ----------
    tensor representing symmetry
    """
    def get_irrep_map(self, sym):
        SYM_INCLUDED=False
        for idx, symi in enumerate(self.sym_lst):
            EQUAL, order = check_sym_equal(symi, sym)
            if EQUAL:
                SYM_INCLUDED=True
                break
        if SYM_INCLUDED:
            if order is None:
                return self.irrep_map_lst[idx]
            else:
                return self.irrep_map_lst[idx].transpose(order)
        else:
            self.update(sym)
            return self.irrep_map_lst[-1]

    make_irrep_map_lst = make_irrep_map_lst

    """
    Return copy of self
    """
    def copy(self):
        newcopy = irrep_map_cache(self.backend)
        newcopy.sym_lst = copy.copy(self.sym_lst)
        newcopy.irrep_map_lst =  copy.copy(self.irrep_map_lst)
        return newcopy
