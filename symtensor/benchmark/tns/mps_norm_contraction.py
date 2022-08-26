#!/usr/bin/env python
import numpy
from symtensor import symlib
from symtensor.sym import random, einsum, core_einsum, zeros
from symtensor.settings import load_lib
import time

def gen_rand_mps(N,D,d,Zn,dZn):
    """
    Returns a random MPS object
    """
    mps = []
    for i in range(N):
        if i == 0:
            sym = ['++-',[range(1),range(dZn),range(Zn)],None,Zn]
            mps.append(random((1,int(d/dZn),int(D/Zn)),sym))
        elif i == N-1:
            sym = ['++-',[range(Zn),range(dZn),range(1)],None,Zn]
            mps.append(random((int(D/Zn),int(d/dZn),1),sym))
        else:
            sym = ['++-',[range(Zn),range(dZn),range(Zn)],None,Zn]
            mps.append(random((int(D/Zn),int(d/dZn),int(D/Zn)),sym))
    # Put into an MPS Object
    mps = MPS(mps)
    return mps

def mpsmps_contract(site1,site2):
    """
    Manually looped contraction of two mps tensors 
    with shared physical bond
    """
    lib = load_lib(site1.backend)
    # Create the resulting tensor
    dZn = len(site1.sym[1][1])
    Zn = len(site1.sym[1][2])
    d = site1.shape[1]
    D = site1.shape[2]
    shape = [D,D]
    sym = ['+-',[range(Zn),range(Zn)],0,Zn]
    res = zeros(shape,sym=sym,backend=site1.backend)
    # Loop through to do contractions
    for i in range(dZn):
        res[i] += lib.einsum('IJ,IK->JK',site1.array[0,i,0],site2.array[0,i,0])
    return res

def envmps1_contract(env,site):
    """
    Manually looped contraction of an MPS tensor
    and a left environment tensor 
    """
    lib = load_lib(env.backend)
    # Create the resulting tensor
    dZn = len(site.sym[1][1])
    Znl = len(site.sym[1][0])
    Znr = len(site.sym[1][2])
    d = site.shape[1]
    Dl = site.shape[0]
    Dr = site.shape[2]
    shape = [Dl,d,Dr]
    sym = ['++-',[range(Znl),range(dZn),range(Znr)],0,Znl]
    res = zeros(shape,sym=sym,backend=site.backend)
    # Loop through to do contractions
    for i in range(Znl):
        j = i
        for k in range(dZn):
            l = (Zn-(j+k))%Zn
            #res[j,k] += lib.einsum('IJ,JKL->IKL',env[i],site[j,k])
            res[j,k] += lib.einsum('JI,JKL->IKL',env[i],site[j,k])
    return res

def envmps2_contract(env,site):
    """
    Manually looped contraction of an MPS/left environment
    tensor with the remaining MPS tensor
    """
    lib = load_lib(env.backend)
    # Create the resulting tensor
    dZn = len(site.sym[1][1])
    Znl= len(site.sym[1][0])
    Znr= len(site.sym[1][2])
    d = site.shape[1]
    Dl= site.shape[0]
    Dr= site.shape[0]
    shape = [Dl,Dr]
    sym = ['+-',[range(Znl),range(Znr)],0,Znl]
    res = zeros(shape,sym=sym,backend=site.backend)
    # Loop through to do contractions
    for i in range(Zn):
        for j in range(dZn):
            k = (Zn-(i+j))%Zn
            l = k
            res[l] += lib.einsum('IJK,IJL->LK',env[i,j],site[i,j])
    return res

# Define an MPS class to do contractions
class MPS:
    """
    Very simple class for a MPS
    """
    def __init__(self,tens):
        """
        Create an MPS with the tensors input as a list
        """
        self.N = len(tens)
        self.tensors = [None]*self.N
        for site in range(self.N):
            self[site] = tens[site].copy()

    def norm(self):
        """
        Contract an mps norm using symtensors
        """
        # Use self as default second tensor
        for i in range(len(self.tensors)):
            print(self.tensors[i].array.shape)
        # Loop through sites contracting the mps ladder
        t0 = time.time()
        for site in range(self.N):
            if site == 0:
                norm_env = einsum('apb,xpy->axby',self[site],self[site])
            else:
                tmp1 = einsum('axby,bpc->axypc',norm_env,self[site])
                norm_env = einsum('axypc,ypz->axcz',tmp1,self[site])
        tf = time.time()
        # Return the resulting norm
        return tf-t0,norm_env.array[0,0,0,0,0,0,0]

    def full_norm(self):
        """
        Contract the norm using full sparse MPS tensors
        """
        # get needed library
        be = load_lib(self[0].backend)
        # Convert tensors into sparse versions
        smps = [self[i].make_sparse() for i in range(self.N)]
        t0 = time.time()
        for site in range(self.N):
            if site == 0:
                norm_env = be.einsum('apbijk,xpyrjs->axbyirks',smps[site],smps[site])
            else:
                tmp1 = be.einsum('axbyirks,bpckjl->axpcyirjls',norm_env,smps[site])
                norm_env = be.einsum('axpcyirjls,ypzsjt->axczirlt',tmp1,smps[site])
        tf = time.time()
        # Return the resulting norm
        return tf-t0,norm_env[0,0,0,0,0,0,0,0]

    def loop_norm(self):
        """
        Contract the norm using symmetric tensors and manual loops over symmetry sectors
        """
        # get needed library
        be = load_lib(self[0].backend)
        t0 = time.time()
        for site in range(self.N):
            if site == 0:
                env = mpsmps_contract(self[site],self[site])
            else:
                tmp1 = envmps1_contract(env,self[site])
                env = envmps2_contract(tmp1,self[site])
        tf = time.time()
        return tf-t0,env[0,0,0]

    def __getitem__(self,i):
        if not hasattr(i,'__len__'):
            return self.tensors[i]
        else:
            return [self.tensors[i[ind]] for ind in range(len(i))]

    def __setitem__(self,site,ten):
        self.tensors[site] = ten

if __name__ == "__main__":
    from sys import argv
    N = int(argv[1])
    D = int(argv[2])
    d = int(argv[3])
    Zn = int(argv[4])
    dZn = Zn#int(argv[5])
    mps1 = gen_rand_mps(N,D,d,Zn,dZn)
    t,norm = mps1.norm()
    print('Symtensor contraction time   = {} (norm = {})'.format(t,norm))
    t,norm = mps1.full_norm()
    print('Full tensor contraction time = {} (norm = {})'.format(t,norm))
    t,norm = mps1.loop_norm()
    print('Loop tensor contraction time = {} (norm = {})'.format(t,norm))
