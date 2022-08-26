import numpy as np
import itertools
import time
from symtensor.sym_blas import tensor, einsum

def compute_cost(s, t, v, G, N):
    maxdim = max(s+t, s+v, t+v) -1
    mem = G**maxdim * N**(maxdim+1) * .8e-8
    scaler = s+t+v-2
    noperations = 1e-9*2.*G**scaler*N**(s+t+v)
    return mem, noperations

def estimate(nops, mem, s, t, v, G):
    N = estimate_from_mem(mem, s, t, v, G)
    mem_, nops_ = compute_cost(s,t,v,G,N)
    return N, mem_, nops_

def estimate_from_flops(nops, s, t, v, G):
    maxdim = max(s+t, s+v, t+v) -1
    scaler = s + t + v - 2
    N = (nops * 1e9/2./(G**scaler))**(1./(s+t+v))
    return int(N)

def estimate_from_mem(mem, s, t, v, G):
    maxdim = max(s+t, s+v, t+v) -1
    N = (mem*1.25e8/(G**maxdim))**(1./(maxdim+1))
    return int(N)

nops = 160 # GFlop count, no effect here
G = 4
mem = 3 # mamimal tensor size, GB

s = t = v = 2 #'ijab,cdab->ijcd'
N, mem_, nops_ = estimate(nops, mem, s, t, v, G)
A = np.random.random([G,G,G,N,N,N,N])
B = np.random.random([G,G,G,N,N,N,N])
sym = ["++--", [range(G)]*4, None, G]
A = tensor(A, sym)
B = tensor(B, sym)
A.symlib.update(sym)
t0 = time.time()
out = einsum('ijab,cdab->ijcd', A, B)
t1 = time.time()

print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))

s = t = v = 3 #'ijkabc,defabc->ijkdef'
N, mem_, nops_ = estimate(nops, mem, s, t, v, G)
A = np.random.random([G,G,G,G,G,N,N,N,N,N,N])
B = np.random.random([G,G,G,G,G,N,N,N,N,N,N])
sym = ["+++---", [range(G)]*6, None, G]
A = tensor(A, sym)
B = tensor(B, sym)
A.symlib.update(sym)
t0 = time.time()
out = einsum('ijkabc,defabc->ijkdef', A, B)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))

s = t = v = 1 #'ia,ab->ib'
N, mem_, nops_ = estimate(nops, mem, s, t, v, G)
A = np.random.random([G,N,N])
B = np.random.random([G,N,N])
sym = ["+-", [range(G)]*2, None, G]
A = tensor(A, sym)
B = tensor(B, sym)
A.symlib.update(sym)
t0 = time.time()
out = einsum('ia,ab->ib', A, B)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))


s, t, v = (1,1,3) #'kdac,ikcd->ia'
N, mem_, nops_ = estimate(nops, mem, s, t, v, G)
A = np.random.random([G,G,G,N,N,N,N])
B = np.random.random([G,G,G,N,N,N,N])
sym = ["++--", [range(G)]*4, None, G]
A = tensor(A, sym)
B = tensor(B, sym)
A.symlib.update(sym)
t0 = time.time()
out = einsum('kdac,ickd->ia', A, B)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))

s, t, v = (1,2,1) #'ab,ijb->ija'
N, mem_, nops = estimate(nops, mem, s, t, v, G)
A = np.random.random([G,N,N])
B = np.random.random([G,G,N,N,N])
sym = ["+-", [range(G)]*2, None, G]
A = tensor(A, sym)
A.symlib.update(sym)
sym = ["++-", [range(G)]*3, None, G]
B = tensor(B, sym)
B.symlib.update(sym)
t0 = time.time()
out = einsum('ab,ijb->ija', A, B)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))

s, t, v = (1,2,2) #'kla,klij->ija'
N, mem_, nops = estimate(nops, mem, s, t, v, G)
A = np.random.random([G,G,N,N,N])
B = np.random.random([G,G,G,N,N,N,N])
sym = ["++-", [range(G)]*3, None, G]
A = tensor(A, sym)
A.symlib.update(sym)
sym = ["++--", [range(G)]*4, None, G]
B = tensor(B, sym)
B.symlib.update(sym)
t0 = time.time()
out = einsum('kla,klij->ija', A, B)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))

s, t, v = (1,3,1) #'ac,ijcb->ijab'
N, mem_, nops = estimate(nops, mem, s, t, v, G)
A = np.random.random([G,N,N])
B = np.random.random([G,G,G,N,N,N,N])
sym = ["+-", [range(G)]*2, None, G]
A = tensor(A, sym)
A.symlib.update(sym)
sym = ["++--", [range(G)]*4, None, G]
B = tensor(B, sym)
B.symlib.update(sym)
t0 = time.time()
out = einsum('ac,ijcb->ijab', A, B)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))

s, t, v = (2,0,2)  #'kcai,kc->ia'
N1 = int((mem*1.25e8/(G**3))**(1./4))
N2 = int((nops*1e9/2./G**2)**(1./4))
N = min(N1, N2)
nops_ = G**2*N**4*1e-9*2.
mem_ = G**3*N**4*.8e-8
A = np.random.random([G,G,G,N,N,N,N])
B = np.random.random([G,N,N])
sym = ["++--", [range(G)]*4, None, G]
A = tensor(A, sym)
A.symlib.update(sym)
sym = ["+-", [range(G)]*2, None, G]
B = tensor(B, sym)
B.symlib.update(sym)
t0 = time.time()
out = einsum('kaci,kc->ia', A, B)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))

s, t, v = (0,0,4)
N1 = int((mem*1.25e8/(G**3))**(1./4))
N2 = int((nops*1e9/2./G**3)**(1./4))
N = min(N1, N2)
nops_ = G**3*N**4*1e-9*2.
mem_ = G**3*N**4*.8e-8
A = np.random.random([G,G,G,N,N,N,N])
B = np.random.random([G,G,G,N,N,N,N])
sym = ["++--", [range(G)]*4, None, G]
A = tensor(A, sym)
B = tensor(B, sym)
A.symlib.update(sym)
out = 0.
t0 = time.time()
out = einsum('ijab,ijab->', A, B)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))

s, t, v = (0,0,2)
N1 = int((mem*1.25e8/(G))**(1./2))
N2 = int((nops*1e9/2./G)**(1./2))
N = min(N1, N2)
nops_ = G*N**2*1e-9*2.
mem_ = G*N**2*.8e-8
A = np.random.random([G,N,N])
B = np.random.random([G,N,N])
sym = ["+-", [range(G)]*2, None, G]
A = tensor(A, sym)
B = tensor(B, sym)
A.symlib.update(sym)
t0 = time.time()
out = einsum('ia,ia->', A, B)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))

s, t, v = (1,0,3)
N1 = int((mem*1.25e8/(G**3))**(1./4))
N2 = int((nops*1e9/2./G**2)**(1./4))
N = min(N1, N2)
nops_ = G**2*N**4*1e-9*2.
mem_ = G**3*N**4*.8e-8
A = np.random.random([G,G,G,N,N,N,N])
B = np.random.random([G,G,N,N,N])
sym = ["++--", [range(G)]*4, None, G]
A = tensor(A, sym)
A.symlib.update(sym)
sym = ["++-", [range(G)]*3, None, G]
B = tensor(B, sym)
B.symlib.update(sym)
t0 = time.time()
out = einsum('lkda,kld->a', A, B)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))


s, t, v = (3,0,1) #'ijcb,c->ijb'
N1 = int((mem*1.25e8/(G**3))**(1./4))
N2 = int((nops*1e9/2./G**2)**(1./4))
N = min(N1, N2)
nops_ = G**2*N**4*1e-9*2.
mem_ = G**3*N**4*.8e-8
A = np.random.random([G,G,G,N,N,N,N])
B = np.random.random([N])
sym = ["++--", [range(G)]*4, None, G]
A = tensor(A, sym)
A.symlib.update(sym)
sym = ["+", [range(G)], None, G]
B = tensor(B, sym)
B.symlib.update(sym)

t0 = time.time()
out = einsum('ijcb,c->ijb', A, B)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))
