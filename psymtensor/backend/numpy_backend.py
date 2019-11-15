import numpy as np

class NUMPYbackend:
    @property
    def name(self):
        return 'numpy'

    def astensor(self, obj, dtype=None):
        if dtype is None:
            return np.asarray(obj)
        else:
            return np.asarray(obj, dtype=dtype)

    def empty(self, shape, dtype=float):
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape, dtype=float):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=float):
        return np.ones(shape, dtype=dtype)

    def copy(self, a):
        return a.copy()

    def rint(self, a):
        return np.rint(a)

    def write(self, a, ind, fill):
        a.put(ind, fill)
        return a

    def to_nparray(self, a):
        return a

    def random(self, size):
        return np.random.random(size)

    def norm(self, a):
        return np.linalg.norm(a)

    def qr(self, a, mode='reduced'):
        return np.linalg.qr(a, mode)

    def dot(self,a,b):
        return np.dot(a,b)

    def __getattr__(self, attr):
        try:
            result = getattr(np, attr)
            return result
        except:
            raise ValueError("Attribute %s not found"%attr)
