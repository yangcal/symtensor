import numpy
import ctf
import symtensor as st
import tensorbackends as tbs

tc = tbs.get("ctf")

# Case 1. input tensor already stored
a = ctf.random.random([2,3,3])
sym = ["+-", [[0,1]]*2, None, 2]
# initalizing by feeding numpy array
symarray = st.array(a, sym, backend=tc)

# Case 2. st.fromfunction to initialize a symtensor object
#------------ Non-symmetric tensors-----------
# st.fromfunction is a wrapper for numpy.st.fromfunction
# returned symtensor object has symtensor.array[i,j] = func(i,j,**kwargs)

sym = None
def per_element(i,j):
    return i+j

shape = (4,4)
symarray = st.fromfunction(per_element, shape, sym=None, backend=tc)

#------------ Symmetric tensors-----------
# st.fromfunction takes a function that generates the numpy tensor block
# from the first N-1 symmetry indices
# the returned symtensor object has symtensor.array[I,J,K] = func(I,J,K,**kwargs)

nbond = 6
G = 3
def per_sym_block(I,J,K):
    """
    generates the block with symmetry indices [I,J,K]
    """
    out = numpy.arange(nbond**4)+I*nbond**2+J*nbond+K
    return out.reshape(nbond,nbond,nbond,nbond)

sym = ["++--", [numpy.arange(G)]*4, None, G]
shape = (nbond,nbond,nbond,nbond)

symarray = st.fromfunction(per_sym_block, shape, sym=sym, backend=tc)
