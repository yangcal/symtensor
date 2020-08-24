import numpy
from symtensor.numpy import array, fromfunction

# Case 1. input tensor already stored
a = numpy.random.random([2,3,3])
sym = ["+-", [[0,1]]*2, None, 2]
# initalizing by feeding numpy array
symarray = array(a, sym)

# Case 2. fromfunction to initialize a symtensor object
#------------ Non-symmetric tensors-----------
# fromfunction is a wrapper for numpy.fromfunction
# returned symtensor object has symtensor.array[i,j] = func(i,j,**kwargs)

sym = None
def per_element(i,j):
    return i+j

shape = (4,4)
symarray = fromfunction(per_element, shape, sym=None)

#------------ Symmetric tensors-----------
# fromfunction takes a function that generates the numpy tensor block
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

symarray = fromfunction(per_sym_block, shape, sym=sym)
