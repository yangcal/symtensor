import ctf
import numpy as np

class CTFbackend:
    @property
    def name(self):
        return 'ctf'

    def astensor(self, obj, dtype=None):
        if dtype is None:
            return ctf.astensor(obj)
        else:
            return ctf.astensor(obj, dtype=dtype)

    def empty(self, shape, dtype=float):
        return ctf.empty(shape, dtype=dtype)

    def zeros(self, shape, dtype=float):
        return ctf.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=float):
        return ctf.ones(shape, dtype=dtype)

    def copy(self, a):
        return a.copy()

    def rint(self, a):
        return ctf.rint(a)

    def write(self, a, ind, fill):
        a.write(ind, fill)
        return a

    def to_nparray(self, a):
        return a.to_nparray()

    def random(self, size):
        return ctf.random.random(size)

    def norm(self, a):
        return a.norm2()

    def qr(self, a):
        return ctf.qr(a)

    def dot(self,a,b):
        return ctf.dot(a,b)

    def write(self, a, ind, fill):
        from psymtensor.tools.mpi_tools import rank
        if rank==0:
            a.write(ind, fill)
        else:
            a.write([],[])

    def __getattr__(self, attr):
        try:
            result = getattr(ctf, attr)
            return result
        except:
            raise ValueError("Attribute %s not found"%attr)
