from symtensor.settings import load_lib
from symtensor.symlib import SYMLIB, fuse_symlib
from symtensor import symlib
from symtensor.misc import DUMMY_STRINGS
from symtensor.tools import utills
import itertools
import numpy as np

BACKEND='numpy'

def get_full_shape(shape, sym):
    if sym is None:
        return shape
    else:
        full_shape = [len(i) for i in sym[1][:-1]] + list(shape)
        return tuple(full_shape)

def zeros(shape, dtype=float, sym=None, backend=BACKEND):
    lib = load_lib(backend)
    full_shape = get_full_shape(shape, sym)
    array = lib.zeros(full_shape, dtype)
    return SYMtensor(array, sym, backend)

def random(shape, sym=None, backend=BACKEND):
    lib = load_lib(backend)
    full_shape = get_full_shape(shape, sym)
    array = lib.random(full_shape)
    tensor = SYMtensor(array, sym, backend)
    tensor.enforce_sym()
    return tensor

def zeros_like(a, dtype=None):
    if dtype is None: dtype=a.dtype
    lib = a.lib
    full_shape = a.array.shape
    array = lib.zeros(full_shape, dtype)
    return a._as_new_tensor(array)

def _transform(Aarray, path, orb_label, lib):
    for ipath in path:
        sym_label, irrep_map = ipath
        subscript = utills.make_subscript(sym_label, [orb_label, '', orb_label], full=False)
        Aarray = lib.einsum(subscript, Aarray, irrep_map)
    return Aarray

def symeinsum(subscripts, op_A, op_B, symlib=None):
    lib = op_A.lib
    if op_A.sym is None:
        # for non-symmetric tensor, no symmetry transformation is needed
        out = lib.einsum(subscripts, op_A, op_B)
        outsym = None
        return SYMtensor(out, outsym, op_A.backend)
    else:
        # two operand contraction supported
        if op_A.sym[3] is not None and op_B.sym[3] is not None:
            # modulus needs to the same for the two operands
            assert(np.allclose(op_A.sym[3], op_B.sym[3]))

        if symlib is None:
            symlib = op_A.symlib + op_B.symlib

        if 'q' in subscripts.lower():
            raise ValueError("q index is reserved for auxillary index, please change the symmetry label in einsum subscript")
        sub_lower= subscripts.lower()

        s_A, s_B, s_C = string_lst = utills.sub_to_lst(sub_lower) # divide into lsts
        sym_string_lst = utills.sub_to_lst(sub_lower.upper())
        out_sym = utills.pre_processing(string_lst, op_A.sym, op_B.sym)
        A, B = op_A.array, op_B.array

        Nind = utills.count_indep_vars(sym_string_lst)
        if utills.is_direct(sym_string_lst, Nind):
            main_subscript = utills.make_subscript(sym_string_lst, string_lst)
            C = lib.einsum(main_subscript, A, B)
        else:
            irrep_map_lst = symlib.make_irrep_map_lst(op_A.sym, op_B.sym, sym_string_lst) # generate all irrep_maps
            A_path, B_path, main_sym_label, C_path = utills.find_path(sym_string_lst, irrep_map_lst, Nind)

            A = _transform(A, A_path, s_A, lib)
            B = _transform(B, B_path, s_B, lib)

            main_subscript = utills.make_subscript(main_sym_label, string_lst, full=False)
            C = lib.einsum(main_subscript, A, B)
            C = _transform(C, C_path, s_C, lib)

        if out_sym is None:
            return C
        else:
            C = SYMtensor(C, out_sym, op_A.backend)
            C.symlib = symlib
            return C

    #else:
    #    raise NotImplementedError

class SYMtensor:
    def __init__(self, array, sym=None, backend=BACKEND):

        assert (len(sym[0])==len(sym[1])),  "sign string length insistent with symmetry range"
        self.array = array
        self.sym = sym
        self.backend = backend
        if sym is None:
            self.symlib = None
            self.ndim = self.array.ndim
            self.shape = self.array.shape
        else:
            self.symlib = SYMLIB(self.backend)
            self.ndim = (self.array.ndim+1) //2
            self.shape  = self.array.shape[self.ndim-1:]

    @property
    def lib(self):
        return load_lib(self.backend)

    @property
    def dtype(self):
        return self.array.dtype

    def norm(self):
        '''compute the Frobenius norm of the tensor'''
        return self.lib.norm(self.array)

    def get_irrep_map(self, sym=None):
        if sym is None: sym=self.sym
        return self.symlib.get_irrep_map(sym)


    def _as_new_tensor(self, x):
        newtensor = SYMtensor(x, self.sym, self.backend)
        return newtensor

    def transpose(self, *axes):
        '''transposing the tensor with specified order'''
        ndim = self.ndim
        assert (len(axes)==ndim), "number of axes does not match the number of symmetry sector"
        if self.sym is None:
            temp = self.array.transpose(axes)
            new_sym = None
        else:
            sign_strings = [self.sym[0][i] for i in axes]
            sym_range = [self.sym[1][i] for i in axes]
            order = list(axes[:ndim-1]) + [i+ndim-1 for i in axes]
            new_sym = [sign_strings, sym_range, self.sym[2], self.sym[3]]
            if axes[-1] == ndim - 1:
                temp = self.array.transpose(tuple(order))
            else:
                s_A = DUMMY_STRINGS[:ndim]
                s_A = DUMMY_STRINGS[:ndim]
                Ain = s_A[:-1].upper() + s_A
                s_A = ''.join([s_A[i] for i in axes])
                Aout = s_A[:-1].upper() + s_A
                irrep_map = self.get_irrep_map()
                sub = Ain + ',' + DUMMY_STRINGS[:ndim].upper() + '->' + Aout
                temp = self.lib.einsum(sub, self.array, irrep_map)
        return SYMtensor(temp, new_sym, self.backend)

    def diagonal(self, preserve_shape=False):
        '''get the diagonal component for tensor with two symmetry sectors, if fill, will return the matrix with diagonal components'''
        if self.ndim ==2 and self.shape[-1]==self.shape[-2]:
            length = self.shape[-1]
            if preserve_shape:
                p = self.lib.eye(length)
                temp = self.lib.einsum('kij,ij->kij',self.array, p)
                return self._as_new_tensor(temp)
            else:
                temp = self.lib.einsum('kii->ki', self.array)
                return temp
        else:
            raise NotImplementedError()

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
        return symlib.get_aux_sym_range(self, idx, phase)

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

    def write_all(self, idx, val):
        self.lib.write_all(self.array, idx, val)

    def write_local(self, idx, val):
        self.lib.write_local(self.array, idx, val)

    def make_sparse(self):
        if self.sym is None: return self.array
        irrep_map  = self.symlib.get_irrep_map(self.sym)
        ndim  = self.ndim
        s_A = DUMMY_STRINGS[:ndim]
        Ain = s_A[:-1].upper() + s_A
        Aout = s_A.upper() + s_A
        sub = Ain + ',' + DUMMY_STRINGS[:ndim].upper() + '->' + Aout
        array = self.lib.einsum(sub, self.array, irrep_map)
        return array

    def enforce_sym(self):
        if self.sym is None: return self
        irrep_map  = self.symlib.get_irrep_map(self.sym)
        ndim  = self.ndim
        s_A = DUMMY_STRINGS[:ndim]
        Ain = s_A.upper() + s_A
        Aout = s_A[:-1].upper() + s_A
        sub = Ain + ',' + DUMMY_STRINGS[:ndim].upper() + '->' + Aout
        sparse = self.make_sparse()
        self.array = self.lib.einsum(sub, sparse, irrep_map)
        sparse = None

tensor = SYMtensor
einsum = symeinsum
core_einsum = np.einsum
