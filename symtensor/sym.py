#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
'''
Symtensor with numpy as backend
'''

import copy
import sys
import numpy as np
import time
from symtensor.settings import load_lib
from symtensor.symlib import *
from symtensor.misc import DUMMY_STRINGS
from symtensor.tools import utills, logger
from symtensor.tools.path import einsum_path

import tensorbackends as tbs

tn = tbs.get('numpy')

def is_symtensor(A):
    return isinstance(A,tensor)

def infer_backend(A):
    if is_symtensor(A):
        return A.array.backend
    elif isinstance(A,np.ndarray):
        return tn
    elif isinstance(A,tbs.interface.Tensor):
        return A.backend
    else:
        print(type(A))
        raise ValueError("SymTensor cannot infer backend")
        

def array(arr, sym=None):
    """
    Create a copy of a symmetric tensor based on tensor data and symmetry

    Parameters
    ----------
    arr: tensor_like
        Input tensor like object

    sym: [string, list(int), int, int]
        Specification of symmetry, refer to main tensor constructor for details

    Returns
    -------
    output: tensor
        A tensor object with specified symmetry.
    """
    return tensor(arr, sym)


def get_full_shape(shape, sym=None):
    """
    Return shape of reduced form of tensor given shape of a symmetry block and the symmetry specification

    Parameters
    ----------
    shape: list(int)
        Shape of symmetric block

    sym: [string, list(int), int, int]
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

def zeros(shape, sym=None, dtype=float, tb=tn):
    """
    Create a zero symmetric tensor with specified symmetry

    Parameters
    ----------
    shape: list(int)
        Shape of each symmetric tensor block

    sym: [string, list(int), int, int]
        Specification of symmetry, refer to main tensor constructor for details

    dtype: type
        Specification of tensor element type

    tb: tensorbackends.backend
        Tensor array backend to use (numpy by default, can be CTF or CuPy)

    Returns
    -------
    output: tensor
        A zeroed out tensor object with specified symmetry.
    """
    full_shape = get_full_shape(shape, sym)
    arr = tn.zeros(full_shape, dtype)
    return array(arr, sym)

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

def random(shape, sym=None, tb=tn):
    full_shape = get_full_shape(shape, sym)
    arr = tn.random.random(full_shape)
    tensor = array(arr, sym)
    tensor.enforce_sym()
    return tensor

def diag(array, sym=None, tb=tn):
    IS_SYMTENSOR = is_symtensor(array)
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
            return tb.diag(array)
        else:
            if array.ndim==3:
                assert(array.shape[1]==array.shape[2])
                out = tb.einsum('kii->ki', array)
                return out
            elif array.ndim==2:
                if not all(len(i)==array.shape[0] for i in sym[1]):
                    raise ValueError("The first dimension of the array must be equal to the number of symmetry sectors")
                raise ValueError("Cannot perform diag with specified parameters")
                #FIXME: code below does not make sense, _sym needs to be defined
                #out = tb.einsum('ki,ij->kij', array, tb.eye(array.shape[-1]))
                #return tensor(out, _sym, tb=tb)
            else:
                raise ValueError("Symmetry not compatible with input array")

def _transform(Aarray, path, orb_label, tb):
    nop = len(path)
    if nop == 0: return Aarray
    for ki, (sym_label, irrep_map) in enumerate(path):
        subscript  = utills.make_subscript(sym_label, [orb_label]+['']*(len(sym_label)-2)+[orb_label], full=False)
        Aarray = tb.einsum(subscript, Aarray, irrep_map)
    return Aarray

def _einsum(subscripts, *operands):
    if len(operands)==1:
        op_A = operands[0]
    else:
        op_A, op_B = operands
    lib = infer_backend(op_A)
    if len(operands)==1:
        if is_symtensor(op_A):
            return lib.einsum(subscripts, op_A.array)
        else:
            return lib.einsum(subscripts, op_A)

    contraction_type = is_symtensor(op_A) + is_symtensor(op_B)
    if contraction_type==0:
        return lib.einsum(subscripts, op_A, op_B)
    elif contraction_type==1:
        if is_symtensor(op_A):
            out = lib.einsum(subscripts, op_A.array, op_B)
            return out
        else:
            out = lib.einsum(subscripts, op_A, op_B.array)
            return out
    cput0 = cput1 = (time.clock(), time.time())
    logger.log(op_A, "Contraction:%s"%subscripts)
    if op_A.sym is None:
        # for non-symmetric tensor, no symmetry transformation is needed
        if '->' not in subscripts:
            subscripts += '->'
        out = lib.einsum(subscripts, op_A.array, op_B.array)
        logger.timer(op_A, "main contraction(non-symmetric) %s"%subscripts, *cput1)
        outsym = None
        if subscripts[-2:]=='->':
            return out
        else:
            return array(out, outsym, backend=op_A.backend)
    else:
        if op_A.sym[3] is not None and op_B.sym[3] is not None:
            # modulus needs to the same for the two operands
            assert(np.allclose(op_A.sym[3], op_B.sym[3]))

        my_symlib = op_A.symlib + op_B.symlib

        if 'q' in subscripts.lower():
            raise ValueError("q index is reserved for auxillary index, please change the symmetry label in einsum subscript")
        sub_lower= subscripts.lower()
        s_A, s_B, s_C = string_lst = utills.sub_to_lst(sub_lower) # divide into lsts
        out_sym = utills.pre_processing(string_lst, op_A.sym, op_B.sym)
        sym_string_lst = utills.sub_to_lst(sub_lower.upper())
        if out_sym is None:
            symbol_lst = [op_A.sym[0], op_B.sym[0], None]
        else:
            symbol_lst = [op_A.sym[0], op_B.sym[0], out_sym[0]]
        for ki, i in enumerate(sym_string_lst):
            sym_string_lst[ki] = utills._cut_non_sym_symbol(i, symbol_lst[ki])
        cput1 = logger.timer_debug(op_A, "pre-processing", *cput1)
        A, B = op_A.array, op_B.array
        Nind = utills.count_indep_vars(sym_string_lst)
        if utills.is_direct(sym_string_lst, Nind):
            main_subscript = utills.make_subscript(sym_string_lst, string_lst)
            C = lib.einsum(main_subscript, A, B)
            logger.timer(op_A, "main contraction %s"%main_subscript, *cput1)
        else:
            bond_dict = {}
            shape = list(op_A.shape) + list(op_B.shape)
            symbols = s_A + s_B
            for ki, i in enumerate(symbols.upper()):
                bond_dict[i] = shape[ki]
            irrep_map_lst = make_irrep_map_lst(my_symlib, op_A._sym, op_B._sym, sym_string_lst) # generate all irrep_maps
            A_path, B_path, main_sym_label, C_path = utills.find_path(sym_string_lst, irrep_map_lst, Nind, bond_dict)
            cput1 = logger.timer_debug(op_A, "finding contraction path", *cput1)
            A = _transform(A, A_path, s_A, lib)
            cput1 = logger.timer_debug(op_A, "transforming input %s"%(s_A), *cput1)
            B = _transform(B, B_path, s_B, lib)
            cput1 = logger.timer_debug(op_A, "transforming input %s"%(s_B), *cput1)
            main_subscript = utills.make_subscript(main_sym_label, string_lst, full=False)
            C = lib.einsum(main_subscript, A, B)
            cput1 = logger.timer(op_A, "main contraction %s"%(main_subscript), *cput1)
            C = _transform(C, C_path, s_C, lib)
            cput1 = logger.timer_debug(op_A, "transforming onput %s"%(s_C), *cput1)

        op_A.symlib = op_B.symlib = my_symlib
        if out_sym is None:
            return C
        else:
            C = tensor(C, out_sym)
            return C

class tensor:
    def __init__(self, array, sym=None, slib=None, tb=tn):
        self.array = array
        self.sym = sym
        self._sym = utills._cut_non_sym_sec(sym)
        if sym is None:
            if slib is None:
              self.symlib = symlib(tb)
            else:
              self.symlib = slib
            self.ndim = self.array.ndim
            self.shape = self.array.shape
        else:
            self.ndim = self.nsym + self.n0sym
            if self.nsym !=0:
                self.shape = self.array.shape[self.nsym-1:]
            else:
                self.shape = self.array.shape
            if slib is None:
                self.symlib = symlib(tb)
            else:
                self.symlib = slib

    def __getattr__(self, item):
        if hasattr(self.array, item):
            return getattr(self.array, item)
        else:
            raise AttributeError("attribute %s not found"%item)

    @property
    def lib(self):
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
        return self.lib.norm(self.array)

    def get_irrep_map(self, sym=None):
        if sym is None: sym=self.sym
        return self.symlib.get_irrep_map(sym)

    def _as_new_tensor(self, x):
        newtensor = tensor(x, self.sym, tb=self.array.backend, \
                          slib=self.symlib)
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
                temp = self.lib.einsum(sub, self.array, irrep_map)
            new_sym = [sign_strings, new_symrange, self.sym[2], self.sym[3]]
        return tensor(temp, new_sym, self.symlib, tb=tb)

    def ravel(self):
        return self.array.ravel()

    def diagonal(self, preserve_shape=False):
        '''get the diagonal component for tensor with two symmetry sectors, if preserve_shape, will return the matrix with diagonal components'''
        if self.n0sym != 0:
            raise NotImplementedError("diagonal not well defined with non-symmetric sector")
        lib = self.lib
        if self.ndim == self.array.ndim:
            if preserve_shape:
                return tensor(lib.diag(lib.diag(self.array)), self.sym, self.symlib, tb=self.array.backend)
            else:
                return lib.diag(self.array)
        elif self.ndim ==2 and self.shape[-1]==self.shape[-2]:
            length = self.shape[-1]
            if preserve_shape:
                p = self.lib.eye(length)
                temp = self.lib.einsum('kij,ij->kij',self.array, p)
                return self._as_new_tensor(temp)
            else:
                temp = self.lib.einsum('kii->ki', self.array)
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
        return symlib.get_aux_sym_range(self.sym, idx, phase)

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
            return array(arr, new_sym, tb=arr.backend)
        else:
            return arr

    def __setitem__(self, key, value):
        self.array[key] = value

    def put(self, idx, val):
        self.lib.put(self.array, idx, val)

    def make_dense(self):
        if self.sym is None: return self.array
        irrep_map  = self.symlib.get_irrep_map(self.sym)
        ndim  = self.ndim
        s_A = DUMMY_STRINGS[:ndim]
        s_A_ = utills._cut_non_sym_symbol(s_A, self.sym[0])
        Ain = s_A_[:-1].upper() + s_A
        Aout = s_A_.upper() + s_A
        sub = Ain + ',' + s_A_.upper() + '->' + Aout
        array = self.lib.einsum(sub, self.array, irrep_map)
        return array

    def enforce_sym(self):
        if self.sym is None: return self
        irrep_map  = self.symlib.get_irrep_map(self.sym)
        ndim  = self.ndim
        s_A = DUMMY_STRINGS[:ndim]
        s_A_ = utills._cut_non_sym_symbol(s_A, self.sym[0])
        Ain = s_A_.upper() + s_A
        Aout = s_A_[:-1].upper() + s_A
        sub = Ain + ',' + s_A_.upper() + '->' + Aout
        sparse = self.make_dense()
        self.array = self.lib.einsum(sub, sparse, irrep_map)
        sparse = None

def einsum(subscripts, *operands):
    newsub = copy.copy(subscripts).replace(' ', '')
    if '->' not in newsub:
        newsub += '->'
    sublist= newsub.replace('->',',').split(',')
    if len(sublist) != len(operands)+1:
        raise ValueError("subscript inputs not matching number of operands")

    use_symmetry = 0
    use_non_symmetry = 0
    for ki, i in enumerate(operands):
        if is_symtensor(i):
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
