
def get(name):
    if name=='numpy':
        from psymtensor.backend.numpy_backend import NUMPYbackend
        return NUMPYbackend()

    elif name=='ctf':
        from psymtensor.backend.ctf_backend import CTFbackend
        return CTFbackend

    else:
        raise ValueError("Backend %s not supported"%name)

