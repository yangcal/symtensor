import numpy as np

BACKEND = 'numpy'

einsum = np.einsum
astensor = np.asarray
zeros = np.zeros
empty = np.empty
ones = np.ones
rint = np.rint
random = np.random.random
norm = np.linalg.norm
qr = np.linalg.qr
dot = np.dot
diag = np.diag

def non_zeros(a):
    idx = np.where(a.ravel()!=0)
    return idx[0]
def copy(a):
    return a.copy()

def write_all(a, ind, fill):
    a.put(ind, fill)
    return a

write_single = write_all

def to_nparray(a):
    return a

def find_less(a, threshold):
    idx = np.where(a.ravel()<threshold)[0]
    return idx
