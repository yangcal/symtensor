#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
'''
Symtensor with numpy as backend
'''

from symtensor.settings import load_lib
from symtensor.symlib import SYMLIB
from symtensor import symlib
from symtensor.misc import DUMMY_STRINGS
from symtensor.tools import utills, logger
from symtensor.tools.path import einsum_path
import sys
import numpy as np
import time

BACKEND='numpy'

__all__ = ["array", "einsum", "zeros", "zeros_like", "diag", "tensor"]

def _unpack_kwargs(**kwargs):
    valid_kwarg = ["backend", "symlib", "verbose", "stdout"]
    for key in kwargs.keys():
        if key not in valid_kwarg:
            raise ValueError("%s not a valid kwarg"%(key))
    _backend = kwargs.get("backend", BACKEND)
    _symlib = kwargs.get("symlib", None)
    _verbose = kwargs.get("verbose", 0)
    _stdout = kwargs.get("stdout", None)
    return _backend, _symlib, _verbose, _stdout

def is_symtensor(tensor):
    return hasattr(tensor, 'array')

def infer_backend(tensor):
    if is_symtensor(tensor):
        return tensor.lib
    else:
        return load_lib(tensor.__class__.__module__.split('.')[0])

def array(arr, sym=None, **kwargs):
    return SYMtensor(arr, sym, **kwargs)

def get_full_shape(shape, sym=None):
    if sym is not None:  assert(len(sym[0])==len(shape))
    if sym is None:
        return shape
    else:
        full_shape = [len(i) for i in sym[1][:-1]] + list(shape)
        return tuple(full_shape)

def zeros(shape, sym=None, dtype=float, **kwargs):
    _backend = _unpack_kwargs(**kwargs)[0]
    lib = load_lib(_backend)
    full_shape = get_full_shape(shape, sym)
    arr = lib.zeros(full_shape, dtype)
    return array(arr, sym, **kwargs)

def zeros_like(a, dtype=None):
    if dtype is None: dtype=a.dtype
    lib = a.lib
    full_shape = a.array.shape
    array = lib.zeros(full_shape, dtype)
    return a._as_new_tensor(array)

def _random(shape, sym=None, **kwargs):
    _backend = _unpack_kwargs(**kwargs)[0]
    lib = load_lib(_backend)
    full_shape = get_full_shape(shape, sym)
    arr = lib.random.random(full_shape)
    tensor = array(arr, sym, **kwargs)
    tensor.enforce_sym()
    return tensor

def diag(array, sym=None, **kwargs):
    _backend, _symlib, _verbose, _stdout = _unpack_kwargs(**kwargs)
    lib = load_lib(_backend)
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
            return lib.diag(array)
        else:
            if array.ndim==3:
                assert(array.shape[1]==array.shape[2])
                out = lib.einsum('kii->ki', array)
                return out
            elif array.ndim==2:
                if not all(len(i)==array.shape[0] for i in sym[1]):
                    raise ValueError("The first dimension of the array must be equal to the number of symmetry sectors")
                out = lib.einsum('ki,ij->kij', array, lib.eye(array.shape[-1]))
                return SYMtensor(out, _sym, symlib=_symlib, verbose=_verbose, stdout=_stdout)
            else:
                raise ValueError("Symmetry not compatible with input array")

def _transform(Aarray, path, orb_label, lib):
    nop = len(path)
    if nop == 0: return Aarray
    symstring, irreps = [], []
    for ki, (sym_label, irrep_map) in enumerate(path):
        if ki ==0: symstring += [sym_label[0]]
        symstring += sym_label[1:-1]
        if ki == nop -1: symstring += [sym_label[-1]]
        irreps.append(irrep_map)
    subscript = utills.make_subscript(symstring, [orb_label]+['']*(len(symstring)-2)+[orb_label], full=False)
    Aarray = lib.einsum(subscript, Aarray, *irreps)
    return Aarray

def pair_einsum(subscripts, op_A, op_B):
    contraction_type = is_symtensor(op_A) + is_symtensor(op_B)
    lib = infer_backend(op_A)
    if contraction_type==0:
        return lib.einsum(subscripts, op_A, op_B)
    elif contraction_type==1:
        if is_symtensor(op_A):
            if getattr(op_A, "sym", None) is None:
                out = lib.einsum(subscripts, op_A.array, op_B)
                return op_A._as_new_tensor(out)
        if is_symtensor(op_B):
            if getattr(op_B, "sym", None) is None:
                out = lib.einsum(subscripts, op_A, op_B.array)
                return op_B._as_new_tensor(out)
        raise TypeError("contraction between symmetric-symtensor and non-symtensor not supported")
    cput0 = cput1 = (time.clock(), time.time())
    verbose = max(op_A.verbose, op_B.verbose)
    op_A.verbose = op_B.verbose = verbose
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
            return array(out, outsym, backend=op_A.backend, verbose=verbose, stdout=op_A.stdout)
    else:
        if op_A.sym[3] is not None and op_B.sym[3] is not None:
            # modulus needs to the same for the two operands
            assert(np.allclose(op_A.sym[3], op_B.sym[3]))

        symlib = op_A.symlib + op_B.symlib

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
            irrep_map_lst = symlib.make_irrep_map_lst(op_A._sym, op_B._sym, sym_string_lst) # generate all irrep_maps
            A_path, B_path, main_sym_label, C_path = utills.find_path(sym_string_lst, irrep_map_lst, Nind, bond_dict)
            cput1 = logger.timer_debug(op_A, "finding contraction path", *cput1)
            A = _transform(A, A_path, s_A, lib)
            cput1 = logger.timer_debug(op_A, "transforming input %s"%s_A, *cput1)
            B = _transform(B, B_path, s_B, lib)
            cput1 = logger.timer_debug(op_A, "transforming input %s"%s_B, *cput1)
            main_subscript = utills.make_subscript(main_sym_label, string_lst, full=False)
            C = lib.einsum(main_subscript, A, B)
            cput1 = logger.timer(op_A, "main contraction %s"%main_subscript, *cput1)
            C = _transform(C, C_path, s_C, lib)
            logger.timer_debug(op_A, "transforming output %s"%s_C, *cput1)

        op_A.symlib = op_B.symlib = symlib
        if out_sym is None:
            return C
        else:
            C = SYMtensor(C, out_sym, op_A.backend, symlib, verbose=verbose, stdout=op_A.stdout)
            return C


class SYMtensor:
    def __init__(self, array, sym=None, backend=BACKEND, symlib=None, verbose=0, stdout=None):
        self.array = array
        self.sym = sym
        self._sym = utills._cut_non_sym_sec(sym)
        self.backend = backend
        if sym is None:
            self.symlib = symlib
            self.ndim = self.array.ndim
            self.shape = self.array.shape
        else:
            self.ndim = self.nsym + self.n0sym
            if self.nsym !=0:
                self.shape = self.array.shape[self.nsym-1:]
            else:
                self.shape = self.array.shape
            if symlib is None:
                self.symlib = SYMLIB(self.backend)
            else:
                self.symlib = symlib
        if stdout is None: stdout = sys.stdout
        self.stdout = stdout
        self.verbose = verbose

    def __getattr__(self, item):
        if hasattr(self.array, item):
            return getattr(self.array, item)
        else:
            raise AttributeError("attribute %s not found"%item)

    @property
    def lib(self):
        return load_lib(self.backend)

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
        newtensor = array(x, self.sym, backend=self.backend, \
                          symlib=self.symlib, verbose=self.verbose, \
                          stdout=self.stdout)
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
        return SYMtensor(temp, new_sym, self.backend, self.symlib, self.verbose, self.stdout)

    def diagonal(self, preserve_shape=False):
        '''get the diagonal component for tensor with two symmetry sectors, if preserve_shape, will return the matrix with diagonal components'''
        if self.n0sym != 0:
            raise NotImplementedError("diagonal not well defined with non-symmetric sector")
        lib = self.lib
        if self.ndim == self.array.ndim:
            if preserve_shape:
                return SYMtensor(lib.diag(lib.diag(self.array)), self.sym, self.backend, self.symlib, self.verbose, self.stdout)
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
        if isinstance(x, SYMtensor):
            return self._as_new_tensor(self.array+x.array)
        else:
            return self._as_new_tensor(self.array+x)

    def __sub__(self, x):
        if isinstance(x, SYMtensor):
            return self._as_new_tensor(self.array-x.array)
        else:
            return self._as_new_tensor(self.array-x)

    def __neg__(self):
        return self._as_new_tensor(-self.array)

    def __mul__(self, x):
        return self._as_new_tensor(self.array*x)

    __rmul__ = __mul__

    def __div__(self, x):
        if isinstance(x, SYMtensor):
            return self._as_new_tensor(self.array/x.array)
        else:
            return self._as_new_tensor(self.array/x)

    __truediv__ = __div__

    def __getitem__(self, key):
        temp = self.array[key]
        ndim = self.ndim

        if self.sym is None:
            if temp.ndim == self.ndim:
                return self._as_new_tensor(temp)
            else:
                return temp
        else:
            if temp.ndim == self.array.ndim and temp.shape[:ndim-1]==self.array.shape[:ndim-1]:
                return self._as_new_tensor(self.array[key])
            else:
                return temp

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


tensor = SYMtensor

def einsum(subscripts, *operands):
    contraction_list = einsum_path(subscripts, *operands)
    operands = [v for v in operands]
    for inds, idx_rm, einsum_str in contraction_list:
        if len(inds) > 2:
            raise ValueError("einsum_path not able to return pairwise contraction, \
                              Please manually perform pairwise contraction")
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str= contraction
        tmp_operands = [operands.pop(x) for x in inds]
        new_view = pair_einsum(einsum_str, tmp_operands[0], tmp_operands[1])
        operands.append(new_view)
        del tmp_operands, new_view
    return operands[0]
