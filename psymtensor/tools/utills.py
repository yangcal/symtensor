import numpy as np
import itertools
import psymtensor.backend as bkd
from psymtensor.tools.params import sign, SYM_TOL
#sign = {'+':1, '-':-1}
#SYM_TOL = 1e-8

def count_indep_vars(string_lst):
    '''Compute the number of independent variables from a list of strings'''
    Nvar = len(set(''.join(string_lst)))
    sorted_string = sorted(string_lst,key=len)
    known_var = set()
    Neq =0
    for i in sorted_string:
        if not set(i).issubset(known_var):
            known_var = known_var | set(i)
            Neq += 1
    Nind = Nvar - Neq
    return Nind

def make_subscript(sym_label, orb_label, full=True):
    ''' generate an einsum subscript from symmetry label and orbital label'''
    ntensor = len(sym_label)
    assert(ntensor==len(orb_label))
    if full: # when all symmetry sectors are specified
        full_list = [sym_label[i][:len(orb_label[i])-1]+orb_label[i] for i in range(ntensor)]
    else:
        full_list = [sym_label[i]+orb_label[i] for i in range(ntensor)]
    return lst_to_sub(full_list)

def lst_to_sub(string_lst):
    ''' convert a string list to einsum subscript'''
    subscript = ','.join(string_lst[:-1]) + '->' + string_lst[-1]
    return subscript

def sub_to_lst(subscript):
    '''convert an einsum subscript to string list'''
    subscript = subscript.replace(' ','')
    if '->' not in subscript: subscript += '->'
    string_lst = subscript.replace('->', ',').split(',')
    return string_lst

def set_to_str(setA):
    '''convert a set of characters to string in sorted order'''
    return ''.join(sorted(setA))

def make_irrep_map(op_A, op_B, sym_label_lst):
    '''generate the delta tensors and strings associated with the symmetry label'''
    s_A, s_B, s_C = sym_label_lst
    contracted = sorted(set(s_A) & set(s_B))
    res_A = set(s_A) - set(contracted)
    res_B = set(s_B) - set(contracted)
    delta_strings = [s_A, s_B]
    delta_tensor = [op_A.get_irrep_map(), op_B.get_irrep_map()]
    if len(contracted) > 1 and len(res_A)>1 and len(res_B)>1:
        # when more than two symmetry sectors are contracted out and the delta does not exist, auxillary index iss generated
        s_q = ''.join(contracted) + 'Q'
        delta_strings.append(s_q)
        idxa = [s_A.find(i) for i in contracted]
        idxb = [s_B.find(i) for i in contracted]
        if op_A.sign_string[idxa[0]] != op_B.sign_string[idxb[0]]:
            phase = -1
        else:
            phase = 1
        auxa = op_A.get_aux_sym_range(idxa)
        auxb = op_B.get_aux_sym_range(idxb, phase)
        aux_range = merge_sym_range(auxa, auxb)
        # when auxillary range is different from the two tensors, pick the shared one for most compact representation
        sign_string = ''.join([op_A.sign_string[i] for i in idxa]) + '-'
        sym_range = [op_A.sym_range[i] for i in idxa]
        sym_range.append(aux_range)
        temp = gen_irrep_map(sign_string, sym_range, op_A.modulus, None, op_A.backend)
        delta_tensor.append(temp)
    delta_lst = fuse_delta([delta_strings, delta_tensor], op_A.backend)
    return delta_lst

def _fuse_delta(sub, delta_a, delta_b, backend):
    """Generate a new delta tensor by tensor contraction between two input delta tensors a and b"""
    if backend.name=='numpy':
        delta = backend.einsum(sub, delta_a, delta_b)
        mask = np.where(delta!=0)
        delta[mask] = 1.0
    elif backend.name=='ctf':
        delta = backend.einsum(sub, delta_a, delta_b)
        [inds, data] = delta.read_local_nnz()
        data[:] = 1.0
        delta = backend.zeros(delta.shape)
        delta.write(inds,data)
    return delta

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
            s_fused = set_to_str((set(s_i)|set(s_j))-ovlp)
            if len(s_fused)==0 or set(s_fused) in string_set:
                continue
            sub = lst_to_sub([s_i,s_j,s_fused])
            delta = _fuse_delta(sub, deltas[i], deltas[j], backend)
            strings.append(s_fused)
            string_set.append(set(s_fused))
            deltas.append(delta)
        ndelta = len(deltas)
    new_strings = sorted(strings,key=len)
    new_deltas = [deltas[strings.index(i)] for i in new_strings]
    return (new_strings, new_deltas)

def is_direct(sym_lst, Nind, full=True):
    """determind if the input sym_label_lst can be contracted directed directly"""
    if full:
        input = set(''.join(i[:-1] for i in sym_lst[:-1]))
        output = set(''.join(sym_lst[-1][:-1]))
    else:
        input = set(''.join(i for i in sym_lst[:-1]))
        output = set(''.join(sym_lst[-1]))
    _direct = (len(input)==Nind and output.issubset(input))
    return _direct

def pre_processing(string_lst, op_A, op_B):
    """compute the output sign_string/rhs shift
    rhs_out = phase * rhs_A + rhs_B
    """
    s_A, s_B, s_C = string_lst
    if s_C == '':
        sign_string = rhs = None
        return sign_string, rhs
    rhs_A, rhs_B = op_A.rhs, op_B.rhs
    contracted = sorted(''.join(set(s_A) & set(s_B)))
    phase = []

    for i in contracted:
        idxa, idxb = s_A.find(i), s_B.find(i)
        syma, symb = op_A.sign_string[idxa], op_B.sign_string[idxb]
        if syma == symb:
            phase.append(-1) #flip the sign of the equation
        else:
            phase.append(1)
    if len(contracted)==0:
        phase = [1]
    sym_correct = (np.unique(phase).size < 2)
    if not sym_correct:
        raise ValueError("SYMMETRY INCOMPATIBLE")
    phase = phase[0]
    rhs = None
    if rhs_A is None:
        rhs = rhs_B
    else:
        rhs = rhs_A * phase
        if rhs_B is not None: rhs += rhs_B
    sign_stringa = op_A.sign_string
    if phase == -1:
        sign_stringa =[]
        for i in op_A.sign_string:
            if i=='+':
                sign_stringa.append('-')
            else:
                sign_stringa.append('+')
    sign_string = ''
    for i in s_C:
        idxa, idxb = s_A.find(i), s_B.find(i)
        if idxa != -1:
            sign_string += sign_stringa[idxa]
        else:
            sign_string += op_B.sign_string[idxb]
    return sign_string, rhs

def enumerate_rep(sym_label, delta_strings, force_out=None):
    """Enumerate all irreducible representation from the given symmetry string and the delta relations"""
    if sym_label =='': return [['', 0, None, 0]]
    Nout = len(set(sym_label))
    delta_set = [set(i) for i in delta_strings]
    for i in delta_strings:
        if set(i).issubset(set(sym_label)):
            Nout -= 1
    if force_out is None: force_out = Nout
    rep = [[sym_label,0,None,0]]
    for ki, i in enumerate(delta_strings):
        s_remain = set(i) - set(sym_label)
        if len(s_remain) > 1:
            continue
        s_all = set(sym_label) | set(i)
        s_shared = set(sym_label) & set(i)
        s_extra = ''.join(s_all - s_shared)
        if len(s_extra) > force_out:
            continue
        else:
            s_shared = ''.join(sorted(s_shared))
            for s_res in itertools.combinations(s_shared, force_out - len(s_extra)):
                s_res = ''.join(sorted(s_res))
                s_out = ''.join(sorted(s_extra + s_res))
                s_leave = s_all - set(s_out)
                if set(s_out) not in delta_set and len(s_leave)<2:
                    rep.append([s_out, 0, ki, 1])
    s_label = [set(i[0]) for i in rep]
    Ncycle = len(rep) - 1
    for i in range(Ncycle):
        s_rep = ''.join(sorted(s_label[i+1]))
        for kj, j in enumerate(delta_strings):
            s_remain = set(j) - set(s_rep)
            if len(s_remain) > 1:
                continue
            s_all = set(s_rep) | set(j)
            s_shared = set(s_rep) & set(j)
            s_extra = ''.join(s_all - s_shared)
            if len(s_extra) > force_out:
                continue
            else:
                s_shared = ''.join(sorted(s_shared))
                for s_res in itertools.combinations(s_shared, force_out - len(s_extra)):
                    s_res = ''.join(sorted(s_res))
                    s_out = ''.join(sorted(s_extra + s_res))
                    s_leave = s_all - set(s_out)
                    if set(s_out) not in s_label and set(s_out) not in delta_set and len(s_leave)<2:
                        s_label.append(set(s_out))
                        if len(j) == 2:
                            rep.append([s_out, i+1, kj, rep[i+1][3]])
                        else:
                            rep.append([s_out, i+1, kj, rep[i+1][3]+1])
    return rep

def make_dict(delta_strings, delta_tensors):
    '''generate the dictionary for estimating cost'''
    dic={}
    for ki,i in enumerate(delta_strings):
        for kj,j in enumerate(i):
            dic[j] = delta_tensors[ki].shape[kj]
    return dic

def estimate_cost(symlist, dic):
    """estimate the cost for the main contraction based on the dimension dictionary"""
    fullstring = set(''.join(symlist))
    fullstring = set_to_str(fullstring)
    count = 1
    for i in fullstring:
        count *= dic[i]
    return count

def find_path(sym_labels, delta_lst, Nind):
    '''find the transformation and contraction path for all tensors'''
    delta_strings, delta_tensors = delta_lst
    dic = make_dict(delta_strings, delta_tensors)
    s_A, s_B, s_C = sym_labels
    Nout = max(len(s_C)-1, 0)
    Ncontracted = Nind - Nout # number of contracted degree of freedom
    a_reps = enumerate_rep(s_A[:-1], delta_strings)
    b_reps = enumerate_rep(s_B[:-1], delta_strings)
    c_reps = enumerate_rep(s_C[:-1], delta_strings)
    tab = []
    counter = []
    cost_tab = []
    for a,b,c in itertools.product(a_reps, b_reps, c_reps):
        good_to_contract = is_direct([a[0],b[0],c[0]], Nind, full=False)
        if not good_to_contract: continue
        A_path = unfold_path(a, a_reps, delta_lst)
        B_path = unfold_path(b, b_reps, delta_lst)
        main_subscript = [a[0],b[0],c[0]]
        C_path = unfold_path(c, c_reps, delta_lst, reverse=True)
        cost = estimate_cost(main_subscript, dic)
        cost_tab.append(cost)
        tab.append([A_path, B_path, main_subscript, C_path])
        counter.append(a[3]+b[3]+c[3])
    idx = counter.index(min(counter))
    nunique = len(set(cost_tab))
    if nunique !=1:
        idx = cost_tab.index(min(cost_tab))
    full_path = tab[idx]
    return full_path

def unfold_path(current_tab, tab, delta_lst, reverse=False):
    '''recover the transformation path from the given position'''
    delta_strings, delta_tensors = delta_lst
    path = []
    while current_tab[2] is not None:
        s_out, delta_idx = current_tab[0], current_tab[2]
        last_tab = tab[current_tab[1]]
        s_in = last_tab[0]
        s_delta, t_delta = delta_strings[delta_idx], delta_tensors[delta_idx]
        current_tab = last_tab
        if reverse:
            subscript = lst_to_sub([s_out, s_delta, s_in])
            path.append([[s_out, s_delta, s_in], t_delta])
        else:
            subscript = lst_to_sub([s_in, s_delta, s_out])
            path.append([[s_in, s_delta, s_out], t_delta])
    if reverse:
        return path
    else:
        return path[::-1]

def gen_irrep_map(sign_string, sym_range, modulus, rhs, backend=bkd.get('numpy')):
    assert(len(sign_string)==len(sym_range))
    if isinstance(sym_range[0][0], int):
        ndim = 1
    else:
        ndim = len(sym_range[0][0])

    def inv(modulus):
        if modulus is None: return None
        if ndim !=1:
            vol = np.linalg.det(modulus)
            modulus_inv = np.asarray([np.cross(modulus[i-2],modulus[i-1])/vol for i in range(ndim)]).T
            modulus_inv = backend.astensor(modulus_inv)
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
            modulus_inv = backend.astensor(inv(modulus))
            val = backend.astensor(val)
            val = backend.dot(val, modulus_inv)
            val = val - backend.rint(val)
            val = backend.to_nparray(val)

    val = np.sum(abs(val), axis=1)
    idx = np.where(val<SYM_TOL)[0]
    fill = np.ones(len(idx))
    shape = [len(i) for i in sym_range]
    delta = np.zeros(shape)
    delta.put(idx, fill)
    return backend.astensor(delta)

def get_aux_sym_range(mytensor, idx, phase=1):
    """compute the range for the auxillary index based on two sides of algebraic equations for the given indices"""
    sym_range, sign_string, rhs, modulus = mytensor.sym_range, mytensor.sign_string, mytensor.rhs, mytensor.modulus
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

    out_left = fold_sym_range(out_left, modulus, mytensor.backend)
    out_right = fold_sym_range(out_right, modulus, mytensor.backend)
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

def fold_sym_range(sym_range, modulus, backend=bkd.get('numpy')):
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
            pbc_inv = backend.astensor(np.asarray([np.cross(modulus[i-2], modulus[i-1])/vol for i in range(ndim)]).T)
            sym_array = backend.astensor(sym_range)
            val = backend.to_nparray(backend.dot(sym_array, pbc_inv))
            val = np.round_(val, 14)
            val = val - np.floor(val)
            val = np.round_(val, 14)
            val = np.unique(val, axis=0)
            val = backend.astensor(val)
            return backend.to_nparray(backend.dot(val, modulus))
