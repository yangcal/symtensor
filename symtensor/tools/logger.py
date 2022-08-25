#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
import sys
import time
VERBOSE = 0
QUIET = 0
TIMER_LEVEL = 1
TIMER_DEBUG = 2

try:
    from mpi4py import MPI
    MPI_FOUND = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except:
    MPI_FOUND=False

if sys.version_info >=(3,6):
    time.clock = time.process_time

def flush(rec, msg, *args):
    rec.stdout.write(msg%args)
    rec.stdout.write('\n')
    rec.stdout.flush()

def log(rec, msg, *args):
    if VERBOSE > QUIET:
        flush(rec, msg, *args)

def timer(rec, msg, cpu0=None, wall0=None):
    if cpu0 is None: cpu0 = rec._t0
    if wall0:
        rec._t0, rec._w0 = time.clock(), time.time()
        if VERBOSE >= TIMER_LEVEL:
            flush(rec, '    CPU time for %s %9.2f sec, wall time %9.2f sec'
                  % (msg, rec._t0-cpu0, rec._w0-wall0))
        return rec._t0, rec._w0

def timer_debug(rec, msg, cpu0=None, wall0=None):
    if cpu0 is None: cpu0 = rec._t0
    if wall0:
        rec._t0, rec._w0 = time.clock(), time.time()
        if VERBOSE >= TIMER_DEBUG:
            flush(rec, '    CPU time for %s %9.2f sec, wall time %9.2f sec'
                  % (msg, rec._t0-cpu0, rec._w0-wall0))
        return rec._t0, rec._w0
