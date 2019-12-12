import ctf
from pyscf.lib import logger
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class Logger(logger.Logger):
    def __init__(self, stdout, verbose):
        if rank == 0:
            logger.Logger.__init__(self, stdout, verbose)
        else:
            logger.Logger.__init__(self, stdout, 0)

def static_partition(tasks):
    segsize = (len(tasks)+size-1) // size
    start = rank * segsize
    stop = min(len(tasks), start+segsize)
    return tasks[start:stop]

NAME='ctf'
einsum = ctf.einsum
astensor = ctf.astensor
zeros = ctf.zeros
empty = ctf.empty
ones = ctf.ones
rint = ctf.rint
random = ctf.random.random

dot = ctf.dot
qr = ctf.qr
diag = ctf.diag
eye = ctf.eye
def non_zeros(a):
    return a.read_all_nnz()[0]

def copy(a):
    return a.copy()

def write_all(a, ind, fill):
    a.write(ind, fill)
    return a

def write_single(a, ind, fill):
    if rank==0:
        a.write(ind, fill)
    else:
        a.write([],[])
    return a

def to_nparray(a):
    return a.to_nparray()

def norm(a):
    return a.norm2()


def find_less(a, threshold):
    c = a.sparsify(threshold)
    idx, vals = a.read_all_nnz()
    return idx
