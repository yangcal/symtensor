from symtensor.settings import load_lib
from symtensor.tools import utills
import itertools
import numpy as np
import copy

SYM_TOL=1e-6
sign = {'+':1,'-':-1}

def sym_to_irrep_map(sym, backend):
    lib = load_lib(backend)
    rank = getattr(lib, "rank", 0)
    sign_string, sym_range, rhs, modulus = sym
    shape = [len(i) for i in sym_range]
    delta = lib.zeros(shape)
    if isinstance(sym_range[0][0], int):
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

        lib.write_all(delta, idx, fill)
    else:
        lib.write_all(delta, [], [])
    return delta


def fold_sym_range(sym_range, modulus):
    """Return the Minimal represetation of a symmetry range"""
    if modulus is None:
        if isinstance(sym_range[0][0], int):
            return np.unique(sym_range)
        else:
            val = np.unique(np.round_(sym_range, 14),axis=0)
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
            val = np.round_(val, 14)
            val = val - np.floor(val)
            val = np.round_(val, 14)
            val = np.unique(val, axis=0)
            return np.dot(val, modulus)

def get_aux_sym_range(sym, idx, phase=1):
    """compute the range for the auxillary index based on two sides of algebraic equations for the given indices"""
    sign_string, sym_range, rhs, modulus = sym
    left_idx, right_idx = idx, [i for i in range(len(sym_range)) if i not in idx]
    nleft, nright = len(left_idx), len(right_idx)
    out_left, out_right = [0,]*2
    if isinstance(sym_range[0][0], int):
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

def merge_sym_range(range_A, range_B):
    delta = abs(np.asarray(range_A)[:,None]-np.asarray(range_B)[None:])
    if delta.ndim !=2:
        delta = np.sum(delta, axis=2)
    idx = np.where(delta<SYM_TOL)
    merged_range = [range_A[i] for i in idx[0]]
    return merged_range

def check_sym_equal(sym1, sym2):
    sign_string1, sym_range1, rhs1, modulus1 = sym1
    sign_string2, sym_range2, rhs2, modulus2 = sym2
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

def fuse_symlib(symlib1, symlib2):
    if symlib1 is None and symlib2 is None:
        return None
    fused_lib = symlib1.copy()
    for ki, i in enumerate(symlib2.sym_lst):
        SYM_INCLUDED = [check_sym_equal(symi, i)[0] for symi in symlib1.sym_lst]
        if not any(SYM_INCLUDED):
            fused_lib.sym_lst.append(i)
            fused_lib.irrep_map_lst.append(symlib2.irrep_map_lst[ki])
    return fused_lib

def make_irrep_map_lst(symlib, sym1, sym2, sym_string_lst):
    sign_string1, sym_range1, rhs1, modulus1 = sym1
    sign_string2, sym_range2, rhs2, modulus2 = sym2
    s_A, s_B, s_C = sym_string_lst

    contracted = sorted(set(s_A) & set(s_B))
    res_A = set(s_A) - set(contracted)
    res_B = set(s_B) - set(contracted)
    delta_strings = [s_A, s_B]
    delta_tensor = [symlib.get_irrep_map(sym1), symlib.get_irrep_map(sym2)]

    if len(contracted) > 1 and len(res_A)>1 and len(res_B)>1:
        # when more than two symmetry sectors are contracted out and the delta does not exist, auxillary index iss generated
        s_q = ''.join(contracted) + 'Q'
        delta_strings.append(s_q)
        idxa = [s_A.find(i) for i in contracted]
        idxb = [s_B.find(i) for i in contracted]
        if sym1[0][idxa[0]] != sym2[0][idxb[0]]:
            phase = -1
        else:
            phase = 1
        auxa = get_aux_sym_range(sym1, idxa)
        auxb = get_aux_sym_range(sym2, idxb, phase)
        aux_range = merge_sym_range(auxa, auxb)
        # when auxillary range is different from the two tensors, pick the shared one for most compact representation
        sign_string = ''.join([sym1[0][i] for i in idxa]) + '-'
        sym_range = [sym1[1][i] for i in idxa]
        sym_range.append(aux_range)
        aux_sym = [sign_string, sym_range, None, sym1[3]]
        delta_tensor.append(symlib.get_irrep_map(aux_sym))
    delta_lst = fuse_delta([delta_strings, delta_tensor], symlib.backend)
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
    lib = load_lib(backend)
    temp = lib.einsum(sub, delta_a, delta_b)
    idx = lib.non_zeros(temp)
    delta = lib.zeros(temp.shape)
    fill = np.ones(len(idx))
    lib.write_single(delta, idx, fill)
    return delta


class SYMLIB:
    def __init__(self, backend):
        self.backend = backend
        self.lib = load_lib(self.backend)
        self.sym_lst = []
        self.irrep_map_lst = []

    def update(self, *sym):
        for i in sym:
            self._update(i)
        return self

    def _update(self, sym):
        SYM_INCLUDED = [self.check_sym_equal(symi, sym)[0] for symi in self.sym_lst]
        if not any(SYM_INCLUDED):
            self.sym_lst.append(sym)
            irrep_map = sym_to_irrep_map(sym, self.backend)
            self.irrep_map_lst.append(irrep_map)

    def __add__(self, symlib2):
        return fuse_symlib(self, symlib2)

    def get_irrep_map(self, sym):
        SYM_INCLUDED=False
        for idx, symi in enumerate(self.sym_lst):
            EQUAL, order = self.check_sym_equal(symi, sym)
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

    def copy(self):
        newcopy = SYMLIB(self.backend)
        newcopy.sym_lst = copy.copy(self.sym_lst)
        newcopy.irrep_map_lst =  copy.copy(self.irrep_map_lst)
        return newcopy

    def check_sym_equal(self, sym1, sym2):
        return check_sym_equal(sym1, sym2)

if __name__=='__main__':
    from pyscf.pbc import gto
    import pyscf.pbc.tools.pbc as tools
    import ctf
    cell = gto.M(a = np.eye(3)*5,atom = '''He 0 0 0''',basis = 'gth-szv',verbose=0)
    kpts = cell.make_kpts([2,2,2]) + np.random.random([1,3])
    gvec = cell.reciprocal_vectors()
    kconserv = tools.get_kconserv(cell, kpts)
    nkpts = len(kpts)
    backend = 'numpy'

    sym1 = ['++--', [kpts,]*4, None, gvec]
    sym2 = ['++-', [kpts,]*3, kpts[0], gvec]
    sym3 = ['+-+-', [kpts,]*4, None, gvec]
    sym4 = ['+-', [kpts,]*2, None, gvec]


    symlib = SYMLIB(backend)
    nocc, nvir = 4, 6
    from symtensor.sym import SYMtensor as tensor
    from symtensor.sym import symeinsum as einsum
    Aarray = np.random.random([nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir])
    Barray = np.random.random([nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir])
    A  = tensor(Aarray, sym1, backend)
    B  = tensor(Barray, sym1, backend)
    C = einsum('ijab,abcd->ijcd', A, B, symlib=symlib)
    Carray = np.zeros([nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir])
    for ki,kj,ka,kc in itertools.product(range(nkpts), repeat=4):
        kb = kconserv[ki,ka,kj]
        kd = kconserv[ki,kc,kj]
        Carray[ki,kj,kc] += np.einsum('ijab,abcd->ijcd', Aarray[ki,kj,ka], Barray[ka,kb,kc])

    print(np.linalg.norm(Carray-C.array))

    #print(C.array.shape)
