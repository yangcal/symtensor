from symtensor.sym import _random
from functools import wraps
def backend_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = 'ctf'
        return func(*args, **kwargs)
    return wrapper

random = backend_wrapper(_random)
