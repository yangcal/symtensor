import unittest
import numpy
from symtensor import symlib
from symtensor.sym import random, einsum, core_einsum




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
        # symtensor contraction
        sym1 = ['++-', [ni,nj,nk], None, None]
        sym2 = ['++-', [nk,nl,nm], None, None]
        ijk = random([nbond,]*3, sym1)
        klm = random([nbond,]*3, sym2)
        ijlm = einsum('ijk,klm->ijlm',ijk,klm)
        # sparse tensor contraction
        ijk_sparse, klm_sparse = ijk.make_sparse(), klm.make_sparse()
        ijlm_sparse = core_einsum('IJKijk,KLMklm->IJLMijlm', ijk_sparse, klm_sparse)
        ijlm_dense = core_einsum('IJLMijlm,IJLM->IJLijlm',ijlm_sparse, ijlm.get_irrep_map())
        diff = (ijlm-ijlm_dense).norm() / numpy.sqrt(ijlm_dense.size)

        self.assertTrue(diff<thresh)

    def test_444(self):
        sym1 = ['+++-', [ni,nj,nl,nm], None, None]
        sym2 = ['++--', [nn,no,nj,nl], None, None]
        ijlm = random([nbond,]*4, sym1)
        nojl = random([nbond,]*4, sym2)
        inom = einsum('ijlm,nojl->inom', ijlm, nojl)
        equal = symlib.check_sym_equal(sym1, sym2)

        ijlm_sparse, nojl_sparse = ijlm.make_sparse(), nojl.make_sparse()
        inom_sparse = core_einsum('IJLMijlm,NOJLnojl->INOMinom', ijlm_sparse, nojl_sparse)
        inom_dense = core_einsum('INOMinom,INOM->INOinom', inom_sparse, inom.get_irrep_map())

        diff = (inom-inom_dense).norm() / numpy.sqrt(inom_dense.size)
        self.assertTrue(diff<thresh)


    def test_433(self):
        sym1 = ['+++-', [ni,nn,no,nm], None, None]
        sym2 = ['++-', [ni,nn,np], None, None]
        inom = random([nbond,]*4, sym1)
        inp = random([nbond,]*3, sym2)
        pom = einsum('inom,inp->pom', inom, inp)

        inom_sparse, inp_sparse = inom.make_sparse(), inp.make_sparse()
        pom_sparse = core_einsum('INOMinom,INPinp->POMpom', inom_sparse, inp_sparse)
        pom_dense = core_einsum('POMpom,POM->POpom', pom_sparse, pom.get_irrep_map())

        diff = (pom -pom_dense).norm()/ numpy.sqrt(pom_dense.size)
        self.assertTrue(diff<thresh)

    def test_453(self):
        ni = range(0,6)
        nj = range(0,3)
        nk = range(1,6)
        nl = range(0,3)
        nm = range(0,3)
        nn = range(0,6)


        sym1 = ['++--', [ni,nj,nk,nl], None, None]
        sym2 = ['-++--', [nj,nk,nl,nm,nn], None, None]

        ijkl = random([nbond,]*4, sym1)
        jklmn = random([nbond,]*5, sym2)
        imn = einsum('ijkl,jklmn->imn', ijkl,jklmn)

        ijkl_sparse, jklmn_sparse = ijkl.make_sparse(), jklmn.make_sparse()
        imn_sparse = core_einsum('IJKLijkl,JKLMNjklmn->IMNimn', ijkl_sparse,jklmn_sparse)
        imn_dense = core_einsum('IMNimn,IMN->IMimn', imn_sparse, imn.get_irrep_map())

        diff = (imn-imn_dense).norm() / numpy.sqrt(imn_dense.size)
        self.assertTrue(diff<thresh)


if __name__ == '__main__':
    unittest.main()
