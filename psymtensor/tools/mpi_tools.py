import time
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
