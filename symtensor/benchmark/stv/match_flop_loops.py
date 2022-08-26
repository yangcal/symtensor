import numpy as np
import itertools
import time

def compute_cost(s, t, v, G, N):
    maxdim = max(s+t, s+v, t+v) -1
    mem = G**maxdim * N**(maxdim+1) * .8e-8
    scaler = s+t+v-2
    noperations = 1e-9*2.*G**scaler*N**(s+t+v)
    return mem, noperations

def estimate(nops, mem, s, t, v, G):
    N = estimate_from_flops(nops, s, t, v, G)
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

nops = 80 # maximal GFlop count
G = 4
mem = 2 # maximal tensor size, no effect here

s = t = v = 2 #'ijab,cdab->ijcd'
N, mem_, nops_ = estimate(nops, mem, s, t, v, G)
A = np.random.random([G,G,G,N,N,N,N])
B = np.random.random([G,G,G,N,N,N,N])
out = np.zeros_like(A)
t0 = time.time()
for ki,kj,ka,kc in itertools.product(range(G), repeat=4):
    kb = np.mod(ki+kj-ka, G)
    kd = np.mod(ki+kj-kc, G)
    out[ki,kj,kc] += np.einsum('ijab,cdab->ijcd', A[ki,kj,ka], B[kc,kd,ka], optimize=True)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))

s = t = v = 3 #'ijkabc,defabc->ijkdef'
N, mem_, nops_ = estimate(nops, mem, s, t, v, G)
A = np.random.random([G,G,G,G,G,N,N,N,N,N,N])
B = np.random.random([G,G,G,G,G,N,N,N,N,N,N])
out = np.zeros_like(A)
t0 = time.time()
for ki,kj,kk,ka,kb,kd,ke in itertools.product(range(G), repeat=7):
    kc = np.mod(ki+kj+kk-ka-kb, G)
    kf = np.mod(ki+kj+kk-kd-ke, G)
    out[ki,kj,kk,kd,ke] += np.einsum('ijkabc,defabc->ijkdef', A[ki,kj,kk,ka,kb], B[kd,ke,kf,ka,kb], optimize=True)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))

s = t = v = 1 #'ia,ab->ib'
N, mem_, nops_ = estimate(nops, mem, s, t, v, G)
A = np.random.random([G,N,N])
B = np.random.random([G,N,N])
out = np.zeros_like(A)
t0 = time.time()
for ki in range(G):
    out[ki] += np.einsum('ia,ab->ib', A[ki], A[ki], optimize=True)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))


s, t, v = (1,1,3) #'kdac,ikcd->ia'
N, mem_, nops_ = estimate(nops, mem, s, t, v, G)
A = np.random.random([G,G,G,N,N,N,N])
B = np.random.random([G,G,G,N,N,N,N])
out = np.zeros([G,N,N])
t0 = time.time()
for kk,kd,ka in itertools.product(range(G), repeat=3):
    kc = np.mod(kk+kd-ka,G)
    out[ka] += np.einsum('kdac,ikcd->ia', A[kk,kd,ka], A[ki,kk,kc], optimize=True)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))

s, t, v = (1,2,1) #'ab,ijb->ija'
N, mem_, nops = estimate(nops, mem, s, t, v, G)
A = np.random.random([G,N,N])
B = np.random.random([G,G,N,N,N])
out = np.zeros_like(B)
t0 = time.time()
for ki,kj in itertools.product(range(G), repeat=2):
    ka = np.mod(ki+kj, G)
    out[ki,kj] = np.einsum('ab,ijb->ija', A[ka], B[ki,kj], optimize=True)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))

s, t, v = (1,2,2) #'kla,klij->ija'
N, mem_, nops = estimate(nops, mem, s, t, v, G)
A = np.random.random([G,G,N,N,N])
B = np.random.random([G,G,G,N,N,N,N])
out = np.zeros_like(A)
t0 = time.time()
for kk,kl,ki in itertools.product(range(G), repeat=3):
    ka = np.mod(kk+kl, G)
    kj = np.mod(kk+kl-ki, G)
    out[ki,kj] = np.einsum('kla,klij->ija', A[kk,kl], B[kk,kl,ki], optimize=True)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))

s, t, v = (1,3,1) #'ac,ijcb->ijab'
N, mem_, nops = estimate(nops, mem, s, t, v, G)
A = np.random.random([G,N,N])
B = np.random.random([G,G,G,N,N,N,N])
out = np.zeros_like(B)
t0 = time.time()
for ki,kj,ka in itertools.product(range(G), repeat=3):
    out[ki,kj,ka] += np.einsum('ac,ijcb->ijab', A[ka], B[ki,kj,ka], optimize=True)
t1 = time.time()
print("(%i,%i,%i),G=%i,N=%i,time=%.2f,mem=%.2f GB, nops=%.2f GFlop, %.1f GFlops"%(s, t, v, G, N, t1-t0, mem_, nops_, nops_/(t1-t0)))
