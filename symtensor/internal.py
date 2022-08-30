#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
'''
Symtensor internal functions and objects
'''

import copy
import sys
import numpy as np
import time
from symtensor.tools import utills, logger
from symtensor.tools.path import einsum_path
import symtensor.tensor
import symtensor.symlib
import tensorbackends as backends

DUMMY_STRINGS='abcdefghijklmnoprstuvwxyz' #q reserved for auxiliary index

tn = backends.get('numpy')
irrep_map_cache_dict = {}

def infer_backend(A):
    """
    Infer backend of a tensor
    """
    if isinstance(A,symtensor.tensor):
        return A.array.backend
    elif isinstance(A,np.ndarray):
        return tn
    elif isinstance(A,backends.interface.Tensor):
        return A.backend
    else:
        print(type(A))
        raise ValueError("SymTensor cannot infer backend")


def _transform(Aarray, path, orb_label, backend):
    """
    Apply irrep map to tensor to align for contraction
    """
    nop = len(path)
    if nop == 0: return Aarray
    for ki, (sym_label, irrep_map) in enumerate(path):
        subscript  = utills.make_subscript(sym_label, [orb_label]+['']*(len(sym_label)-2)+[orb_label], full=False)
        Aarray = backend.einsum(subscript, Aarray, irrep_map)
    return Aarray

def _einsum(subscripts, *operands):
    """
    Perform 2-tensor contraction of symtensors by aligning symmetries via transformng the irreps
    """
    if len(operands)==1:
        op_A = operands[0]
    else:
        op_A, op_B = operands
    lib = infer_backend(op_A)
    if len(operands)==1:
        if isinstance(op_A,symtensor.tensor):
            return lib.einsum(subscripts, op_A.array)
        else:
            return lib.einsum(subscripts, op_A)

    contraction_type = isinstance(op_A,symtensor.tensor) + isinstance(op_B,symtensor.tensor)
    if contraction_type==0:
        return lib.einsum(subscripts, op_A, op_B)
    elif contraction_type==1:
        if isinstance(op_A,symtensor.tensor):
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

        assert(op_A.irrep_map_cache is op_B.irrep_map_cache)
        my_irrep_map_cache = op_A.irrep_map_cache

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
            irrep_map_lst = symtensor.symlib.make_irrep_map_lst(my_irrep_map_cache, op_A._sym, op_B._sym, sym_string_lst) # generate all irrep_maps
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

        op_A.irrep_map_cache = op_B.irrep_map_cache = my_irrep_map_cache
        if out_sym is None:
            return C
        else:
            C = symtensor.tensor(C, out_sym)
            return C


