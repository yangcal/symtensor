import numpy as np
import itertools
sign = {'+':1, '-':-1}
SYM_TOL = 1e-8

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

def pre_processing(string_lst, symA, symB):
    """compute the output symmetry from the contraction of symA and symB
    """
    s_A, s_B, s_C = string_lst
    if s_C == '':
        return None
    rhs_A, rhs_B = symA[2], symB[2]
    if rhs_A is None: rhs_A = 0
    if rhs_B is None: rhs_B  = 0
    contracted = sorted(''.join(set(s_A) & set(s_B)))
    phase = []
    FLIP = {'+':'-','-':'+'}
    for i in contracted:
        idxa, idxb = s_A.find(i), s_B.find(i)
        if symA[0][idxa]==symB[0][idxb]:
            phase.append(-1) #flip the sign of the equation
        else:
            phase.append(1)
    if len(contracted)==0:
        phase = [1]
    sym_correct = (np.unique(phase).size < 2)
    if not sym_correct:
        raise ValueError("SYMMETRY INCOMPATIBLE")
    phase = phase[0]
    rhs = rhs_A * phase + rhs_B
    if phase == 1:
        sign_stringa = symA[0]
    else:
        sign_stringa = ''.join([FLIP[i] for i in symA[0]])
    sign_string = ''
    sym_range = []
    for i in s_C:
        idxa, idxb = s_A.find(i), s_B.find(i)
        if idxa != -1:
            sign_string += sign_stringa[idxa]
            sym_range.append(symA[1][idxa])
        else:
            sign_string += symB[0][idxb]
            sym_range.append(symB[1][idxb])
    out_sym = (sign_string, sym_range, rhs, symA[3])
    return out_sym

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
