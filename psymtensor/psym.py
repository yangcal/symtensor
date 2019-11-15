import numpy as np
import backend as bkd
import itertools
from psymtensor.tools import utills
from psymtensor.tools.params import DUMMY_STRINGS

def einsum(subscripts, op_A, op_B):
    r'''Main wrapper to perform contraction with periodic symmetry, only contraction between two psymtensors supported

    Args:
        subscripts :
            einsum subscript
        op_A, op_B :
            PSYMtensor objects

    Returns:
        value or contracted PSYMtensor object
    '''
    assert(op_A.backend.name == op_B.backend.name)

    if op_A.modulus is not None and op_B.modulus is not None:
        assert(np.allclose(op_A.modulus, op_B.modulus))

    if 'q' in subscripts.lower():
        raise ValueError("q index is reserved for auxillary index, please change the symmetry label")

    backend = op_A.backend

    subscripts= subscripts.lower()
    s_A, s_B, s_C = string_lst = utills.sub_to_lst(subscripts) # divide into lsts
    sym_string_lst = utills.sub_to_lst(subscripts.upper())
    out_sign_string, out_rhs = utills.pre_processing(string_lst, op_A, op_B)

    A, B = op_A.array, op_B.array
    Nind = utills.count_indep_vars(sym_string_lst)
    if utills.is_direct(sym_string_lst, Nind):
        main_subscript = utills.make_subscript(sym_string_lst, string_lst)
        C = backend.einsum(main_subscript, A, B)
    else:
        irrep_map_lst = utills.make_irrep_map(op_A, op_B, sym_string_lst) # generate all the irrep_map tensors needed
        A_path, B_path, main_sym_label, C_path = utills.find_path(sym_string_lst, irrep_map_lst, Nind) # figure out the path for contraction
        for a in A_path:
            sym_label, P = a
            sub_A = utills.make_subscript(sym_label, [s_A,'',s_A], full=False)
            if op_A.verbose>0:
                print("Transforming A by: %s"%sub_A, P.shape)
            A = backend.einsum(sub_A, A, P)
        for b in B_path:
            sym_label, P = b
            sub_B = utills.make_subscript(sym_label, [s_B,'',s_B], full=False)
            if op_B.verbose>0:
                print("Transforming B by: %s"%sub_B, P.shape)
            B = backend.einsum(sub_B, B, P)
        main_subscript = utills.make_subscript(main_sym_label, string_lst, full=False)
        if op_A.verbose>0:
            print("Main Contraction Step: %s"%main_subscript)
        C = backend.einsum(main_subscript, A, B)
        for c in C_path:
            sym_label, P = c
            sub_C = utills.make_subscript(sym_label, [s_C,'',s_C], full=False)
            if op_A.verbose>0:
                print("Transforming OUTPUT by: %s"%sub_C)
            C = backend.einsum(sub_C, C, P)
    if out_sign_string is None:
        return C
    else:
        out_range = []
        for i in s_C:
            if i in s_A:
                idx = s_A.index(i)
                out_range.append(op_A.sym_range[idx])
            else:
                idx = s_B.index(i)
                out_range.append(op_B.sym_range[idx])
        outmodulus = op_A.modulus
        if op_A.modulus is None:
            outmodulus= op_B.modulus
        return PSYMtensor(out_sign_string, out_range, None, op_A._backend, outmodulus, out_rhs, None, C, op_A.verbose)


class PSYMtensor:
    '''Basic class for tensors with symmetry information

    Attributes:
        sign_string : str
            algebraic sign_strings denoting symmetry relations, e.g. "+-++", so that symmetry sector s_{ijkl} is nonzero if
            symrange[i] - symrange[j] + symrange[k] + symrange[l] = rhs (mod modulus)
        sym_range : list
            a list of possible values for each index into the symmetry sector, eg. [[0,1,2],[2,3,5],[3,5,6]]
        shape: tuple
            the shape of each tensor block / symmetry sector
        backend: np or ctf
            the tensor library to use
        modulus : int or array or None
            the periodic boundary condition enforced on the algebraic equation (by computing modulo modulus)
        rhs : int or array
            the right hand side of the algebraic equation (remainer of modulus operation or overall result if modulus = None)
        dtype:
            the data type of the tensor
        array : np or ctf array
            dense array with order 2n-1 where n=len(sign_string), specify alternative to shape/dtype
    '''
    def __init__(self, sign_string, sym_range, shape=None, backend="numpy", modulus=None, rhs=None, dtype=None, array=None, verbose=0):

        assert (len(sign_string)==len(sym_range)),  "sign string length insistent with symmetry range"

        self.sign_string = sign_string
        self.sym_range = sym_range
        self.shape = shape
        self._backend = backend
        self.modulus = modulus
        self.rhs = rhs
        self.ndim = len(sym_range)
        if dtype is None:
            self.dtype = np.float64
        else:
            self.dtype = dtype

        if array is not None:
            ndim = self.ndim
            if shape is None:
                self.shape = array.shape[ndim-1:]
            else:
                assert (array.shape[ndim-1:]==tuple(shape)),  "shape insistent with the input array"

        self.array = array
        self.verbose  = verbose

    @property
    def backend(self):
        return bkd.get(self._backend)

    def norm(self):
        '''compute the Frobenius norm of the tensor'''
        return self.backend.norm(self.array)

    def get_irrep_map(self, idx=None, rhs=None, aux=False):
        '''get the irrep_map tensor for the correspoding symmetry sector'''
        if idx is None: idx = range(self.ndim)
        if rhs is None: rhs = self.rhs
        sign_string = [self.sign_string[i] for i in idx]
        sym_range = [self.sym_range[i] for i in idx]
        modulus = self.modulus
        if aux:
            aux_range = self.find_aux_range(idx)
            sign_string += '-'
            sym_range.append(aux_range)
        return utills.gen_irrep_map(sign_string, sym_range, modulus, rhs, self.backend)

    def _as_new_tensor(self, x):
        return PSYMtensor(self.sign_string, self.sym_range, None, self._backend, self.modulus, self.rhs, self.dtype, x)


    def transpose(self, *axes):
        '''transposing the tensor with specified order'''
        ndim = self.ndim
        assert (len(axes)==ndim), "number of axes does not match the number of symmetry sector"
        shape = [self.shape[i] for i in axes]
        order = list(axes[:ndim-1]) + [i+ndim-1 for i in axes]
        sign_string = [self.sign_string[i] for i in axes]
        sym_range = [self.sym_range[i] for i in axes]

        if axes[-1] == ndim-1:
            temp = self.array.transpose(tuple(order))
        else:
            # contraction with irrep_map tensor needed
            s_A = DUMMY_STRINGS[:ndim]
            Ain = s_A[:-1].upper() + s_A
            s_A = ''.join([s_A[i] for i in axes])
            Aout = s_A[:-1].upper() + s_A
            irrep_map = self.get_irrep_map()
            sub = Ain + ',' + DUMMY_STRINGS[:ndim].upper() + '->' + Aout
            temp = self.backend.einsum(sub, self.array, irrep_map)
        return PSYMtensor(sign_string, sym_range, shape, self._backend, self.modulus, self.rhs, self.dtype, temp)

    def diagonal(self, preserve_shape=False):
        '''get the diagonal component for tensor with two symmetry sectors, if fill, will return the matrix with diagonal components'''
        if self.ndim ==2 and self.shape[-1]==self.shape[-2]:
            length = self.shape[-1]
            if preserve_shape:
                p = self.backend.eye(length)
                temp = self.backend.einsum('kij,ij->kij',self.array, p)
                return self._as_new_tensor(temp)
            else:
                temp = backend.einsum('kii->ki', self.array)
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
        return utills.get_aux_sym_range(self, idx, phase)

    def copy(self):
        return self._as_new_tensor(self.array)

    def conj(self):
        """compute conjugate of the array"""
        return self._as_new_tensor(self.array.conj())

    def __add__(self, x):
        if isinstance(x, PSYMtensor):
            return self._as_new_tensor(self.array+x.array)
        else:
            return self._as_new_tensor(self.array+x)

    def __sub__(self, x):
        if isinstance(x, PSYMtensor):
            return self._as_new_tensor(self.array-x.array)
        else:
            return self._as_new_tensor(self.array-x)

    def __neg__(self):
        return self._as_new_tensor(-self.array)

    def __mul__(self, x):
        return self._as_new_tensor(self.array*x)

    __rmul__ = __mul__

    def __div__(self, x):
        if isinstance(x, PSYMtensor):
            return self._as_new_tensor(self.array/x.array)
        else:
            return self._as_new_tensor(self.array/x)

    __truediv__ = __div__

    def __getitem__(self, key):
        temp = self.array[key]
        if temp.ndim == self.array.ndim:
            return self._as_new_tensor(self.array[key])
        else:
            return temp

    def __setitem__(self, key, value):
        self.array[key] = value

    def write(self, idx, val):
        if self._backend == 'numpy':
            self.array.put(idx, val)
        else:
            self.array.write(idx, val)

    def make_sparse(self):
        irrep_map = self.get_irrep_map()
        ndim  = self.ndim
        s_A = DUMMY_STRINGS[:ndim]
        Ain = s_A[:-1].upper() + s_A
        Aout = s_A.upper() + s_A
        sub = Ain + ',' + DUMMY_STRINGS[:ndim].upper() + '->' + Aout
        sparse = self.backend.einsum(sub, self.array, irrep_map)
        return sparse

    def enforce_sym(self):
        irrep_map  = self.get_irrep_map()
        ndim  = self.ndim
        s_A = DUMMY_STRINGS[:ndim]
        Ain = s_A.upper() + s_A
        Aout = s_A[:-1].upper() + s_A
        sub = Ain + ',' + DUMMY_STRINGS[:ndim].upper() + '->' + Aout
        sparse = self.make_sparse()
        self.array = self.backend.einsum(sub, sparse, irrep_map)
        sparse = None

def zeros(sign_string, sym_range, shape, backend='numpy', modulus=None, rhs=None, dtype=float, verbose=0):
    '''FIXME: match order of params appearing in __init__ above'''
    if len(shape) == len(sign_string):
        full_shape = [len(i) for i in sym_range[:-1]] + list(shape)
    elif len(shape) == 2*len(sign_string)-1:
        full_shape = shape
    else:
        raise ValueError("Shape inconsistent with symmetry sector")
    func = bkd.get(backend)
    dat = func.zeros(full_shape, dtype=dtype)
    return PSYMtensor(sign_string, sym_range, None, backend, modulus, rhs, dtype, dat, verbose)

def random(sign_string, sym_range, shape, backend='numpy', modulus=None, rhs=None, verbose=0):
    '''FIXME: match order of params appearing in __init__ above'''
    if len(shape) == len(sign_string):
        full_shape = [len(i) for i in sym_range[:-1]] + list(shape)
    elif len(shape) == 2*len(sign_string)-1:
        full_shape = shape
    else:
        raise ValueError("Shape inconsistent with symmetry sector")
    func = bkd.get(backend)
    dat = func.random(full_shape)
    rand = PSYMtensor(sign_string, sym_range, None, backend, modulus, rhs, None, dat, verbose) #enforce symmetry in rand
    rand.enforce_sym()
    return rand

if __name__=='__main__':
    a=0
