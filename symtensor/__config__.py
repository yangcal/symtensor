BACKEND_DIC = {'numpy'}
DEFAULT_BACKEND = 'numpy'

try:
    import ctf
    WITH_CTF = True
    BACKEND_DIC.add('ctf')
except:
    WITH_CTF = False

try:
    import cupy
    WITH_CUPY = True
    BACKEND_DIC.add('cupy')
except:
    WITH_CUPY = False

if DEFAULT_BACKEND not in BACKEND_DIC:
    raise ValueError("Backend %s not found"%(DEFAULT_BACKEND))
