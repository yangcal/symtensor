#!/usr/bin/env python
import numpy
from symtensor import symlib
from symtensor.sym import random, einsum, core_einsum
from symtensor.settings import load_lib
import time

def rand_bmps(D,chi,Zn,backend):
    """
    Create a boundary mps tensor
    """
    return random((chi/Zn,D/Zn,chi/Zn),['++-',[range(Zn),range(Zn),range(Zn)],0,Zn],backend=backend)

def rand_peps(d,D,chi,Zn,backend):
    """
    Create a standard PEPS tensor
    """
    return random((D/Zn,D/Zn,d/Zn,D/Zn,D/Zn),['+++--',[range(Zn),range(Zn),range(Zn),range(Zn),range(Zn)],0,Zn],backend=backend)

def contract_symtensor(bmps,peps):
    """
    Contract the bmps and peps using symtensors
    """
    t0 = time.time()
    res = einsum('mln,ldpru->mndpru',bmps,peps)
    tf = time.time()
    return tf-t0,res

def contract_full(bmps,peps):
    """
    Contract the bmps and peps using full sparse tensors
    """
    backend = load_lib(bmps.backend)
    # Convert to full tensors
    fbmps = bmps.make_sparse()
    fpeps = peps.make_sparse()
    # Do timed contraction with full tensors
    t0 = time.time()
    res = backend.einsum('mlnMLN,ldpruLDPRU->mndpruMNDPRU',fbmps,fpeps)
    tf = time.time()
    return tf-t0,res

def contract_looped(bmps,peps,d,D,chi,Zn):
    """
    Contract the bmps and peps using manual loops over symmetry sectors
    """
    backend = load_lib(bmps.backend)
    # Get set up 
    res = backend.zeros((Zn,Zn,Zn,Zn,Zn,chi/Zn,chi/Zn,D/Zn,d/Zn,D/Zn,D/Zn))
    # Do timed contraction with loops over symmetry sectors
    t0 = time.time()
    for m in range(Zn):
        for n in range(Zn):
            l1 = (m-n)%Zn
            l = (Zn-(m-n))%Zn
            for d in range(Zn):
                for p in range(Zn):
                    for r in range(Zn):
                        u = (Zn-(-l-d-p+r))%Zn
                        res[m,n,d,p,r] = backend.einsum('MLN,LDPRU->MNDPRU',bmps[m,l],peps[l,d,p,r])
    tf = time.time()
    return tf-t0,res

def run_test(d,D,chi,Zn,Ncalc,backend,debug=False):
    be = load_lib(backend)
    # Loop over all required calculations
    tsym  = [None]*Ncalc
    tfull = [None]*Ncalc
    tloop = [None]*Ncalc
    for icalc in range(Ncalc):

        # Create the tensors
        bmps = rand_bmps(D,chi,Zn,backend)
        peps = rand_peps(d,D,chi,Zn,backend)

        # Do contraction with symtensors
        tsym[icalc],ressym = contract_symtensor(bmps,peps)

        # Do contraction with full matrices
        tfull[icalc],resfull = contract_full(bmps,peps)
    
        # Do contraction via loops over symmetry sectors
        tloop[icalc],resloop = contract_looped(bmps,peps,d,D,chi,Zn)

    return sum(tsym)/Ncalc,sum(tfull)/Ncalc,sum(tloop)/Ncalc

if __name__ == "__main__":
    from sys import argv
    d = int(argv[1])
    D = int(argv[2])
    chi = int(argv[3])
    Zn = int(argv[4])
    ctf = int(argv[5])
    if ctf == 0:
        backend = 'numpy'
    else:
        backend = 'ctf'
    Ncalc = 10
    tsym,tfull,tloop = run_test(d,D,chi,Zn,Ncalc,backend)
    print('Average times are:')
    print('\tSymtensor time {}'.format(tsym))
    print('\tFull time {}'.format(tfull))
    print('\tLooped time {}'.format(tloop))
    print('With Parameters:')
    print('\tD   = {}'.format(D))
    print('\td   = {}'.format(d))
    print('\tchi = {}'.format(chi))
    print('\tZn  = {}'.format(Zn))
