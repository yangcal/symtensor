import unittest
import numpy
from pyscf.pbc import gto
import pyscf.pbc.tools.pbc as tools
import psymtensor.psymtensor as tsn
import itertools

tensor = tsn.PSYMtensor
einsum = tsn.einsum
random = tsn.random
backend = 'ctf'
backend = 'numpy'
if backend == 'numpy':
    bkn = numpy
elif backend=='ctf':
    import ctf
    bkn = ctf

thresh = 1e-8
sign = {'+':1, '-':-1}


nbond  = 8
ni = range(0,5)
nj = range(0,2)
nk = range(1,5)
nl = range(0,4)
nm = range(1,5)
nn = range(0,5)
no = range(0,2)
np = range(0,5)


class DMRGnpTEST(unittest.TestCase):
    def test_334(self):
        ijk = random('++-', [ni,nj,nk], [nbond,]*3, backend, None, None)
        klm = random('++-', [nk,nl,nm], [nbond,]*3, backend, None, None)
        ijk_sparse, klm_sparse = ijk.make_sparse(), klm.make_sparse()

        ijlm = einsum('ijk,klm->ijlm',ijk,klm)

        ijlm_sparse = bkn.einsum('IJKijk,KLMklm->IJLMijlm', ijk_sparse, klm_sparse)
        ijlm_dense = bkn.einsum('IJLMijlm,IJLM->IJLijlm',ijlm_sparse,ijlm.get_irrep_map())
        diff = (ijlm-ijlm_dense).norm() / numpy.sqrt(ijlm_dense.size)
        self.assertTrue(diff<thresh)

    def test_444(self):

        ijlm = random('+++-', [ni,nj,nl,nm], [nbond,]*4, backend, None, None)
        nojl = random('++--', [nn,no,nj,nl], [nbond,]*4, backend, None, None)
        ijlm_sparse, nojl_sparse = ijlm.make_sparse(), nojl.make_sparse()


        inom = einsum('ijlm,nojl->inom', ijlm, nojl)
        inom_sparse = bkn.einsum('IJLMijlm,NOJLnojl->INOMinom', ijlm_sparse, nojl_sparse)
        inom_dense = bkn.einsum('INOMinom,INOM->INOinom', inom_sparse, inom.get_irrep_map())

        diff = (inom-inom_dense).norm() / numpy.sqrt(inom_dense.size)
        self.assertTrue(diff<thresh)


    def test_433(self):
        inom = random('+++-', [ni,nn,no,nm], [nbond,]*4, backend, None, None)
        inp = random('++-', [ni,nn,np], [nbond,]*3, backend, None, None)
        inom_sparse, inp_sparse = inom.make_sparse(), inp.make_sparse()

        pom = einsum('inom,inp->pom', inom, inp)
        pom_sparse = bkn.einsum('INOMinom,INPinp->POMpom', inom_sparse, inp_sparse)
        pom_dense = bkn.einsum('POMpom,POM->POpom', pom_sparse, pom.get_irrep_map())

        diff = (pom -pom_dense).norm()/ numpy.sqrt(pom_dense.size)
        self.assertTrue(diff<thresh)

    def test_453(self):
        ni = range(0,6)
        nj = range(0,3)
        nk = range(1,6)
        nl = range(0,3)
        nm = range(0,3)
        nn = range(0,6)

        ijkl = random('++--', [ni,nj,nk,nl], [nbond,]*4, backend, None, None)
        jklmn = random('-++--', [nj,nk,nl,nm,nn], [nbond,]*5, backend, None, None)
        ijkl_sparse, jklmn_sparse = ijkl.make_sparse(), jklmn.make_sparse()

        imn = einsum('ijkl,jklmn->imn', ijkl,jklmn)
        imn_sparse = bkn.einsum('IJKLijkl,JKLMNjklmn->IMNimn', ijkl_sparse,jklmn_sparse)
        imn_dense = bkn.einsum('IMNimn,IMN->IMimn', imn_sparse, imn.get_irrep_map())

        diff = (imn-imn_dense).norm() / numpy.sqrt(imn_dense.size)
        self.assertTrue(diff<thresh)


if __name__ == '__main__':
    unittest.main()
