#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

import unittest
import numpy
from symtensor import symlib, random, einsum

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
        ijk = random.random([nbond,]*3, sym1)
        klm = random.random([nbond,]*3, sym2)
        ijlm = einsum('ijk,klm->ijlm',ijk,klm)
        # sparse tensor contraction
        ijk_dense, klm_dense = ijk.make_dense(), klm.make_dense()
        ijlm_dense = einsum('IJKijk,KLMklm->IJLMijlm', ijk_dense, klm_dense)
        ijlm_sparse = einsum('IJLMijlm,IJLM->IJLijlm',ijlm_dense, ijlm.get_irrep_map())
        diff = (ijlm-ijlm_sparse).norm() / numpy.sqrt(ijlm_sparse.size)
        self.assertTrue(diff<thresh)

    def test_444(self):
        numpy.random.seed(3)
        sym1 = ['+++-', [ni,nj,nl,nm], None, None]
        sym2 = ['++--', [nn,no,nj,nl], None, None]
        ijlm = random.random([nbond,]*4, sym1)
        nojl = random.random([nbond,]*4, sym2)
        inom = einsum('ijlm,nojl->inom', ijlm, nojl)
        equal = symlib.check_sym_equal(sym1, sym2)

        ijlm_dense, nojl_dense = ijlm.make_dense(), nojl.make_dense()
        inom_dense = einsum('IJLMijlm,NOJLnojl->INOMinom', ijlm_dense, nojl_dense)
        inom_sparse = einsum('INOMinom,INOM->INOinom', inom_dense, inom.get_irrep_map())
        diff = (inom-inom_sparse).norm() / numpy.sqrt(inom_sparse.size)
        self.assertTrue(diff<thresh)


    def test_433(self):
        sym1 = ['+++-', [ni,nn,no,nm], None, None]
        sym2 = ['++-', [ni,nn,np], None, None]
        inom = random.random([nbond,]*4, sym1)
        inp = random.random([nbond,]*3, sym2)
        pom = einsum('inom,inp->pom', inom, inp)

        inom_dense, inp_dense = inom.make_dense(), inp.make_dense()
        pom_dense = einsum('INOMinom,INPinp->POMpom', inom_dense, inp_dense)
        pom_sparse = einsum('POMpom,POM->POpom', pom_dense, pom.get_irrep_map())

        diff = (pom -pom_sparse).norm()/ numpy.sqrt(pom_sparse.size)
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

        ijkl = random.random([nbond,]*4, sym1)
        jklmn = random.random([nbond,]*5, sym2)
        imn = einsum('ijkl,jklmn->imn', ijkl,jklmn)

        ijkl_dense, jklmn_dense = ijkl.make_dense(), jklmn.make_dense()
        imn_dense = einsum('IJKLijkl,JKLMNjklmn->IMNimn', ijkl_dense,jklmn_dense)
        imn_sparse = einsum('IMNimn,IMN->IMimn', imn_dense, imn.get_irrep_map())

        diff = (imn-imn_sparse).norm() / numpy.sqrt(imn_sparse.size)
        self.assertTrue(diff<thresh)

if __name__ == '__main__':
    unittest.main()
