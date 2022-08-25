#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

import unittest
import numpy
import symtensor as st
from symtensor.symlib import check_sym_equal
import tensorbackends as tbs

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

tb = tbs.get('numpy')


class DMRGnpTEST(unittest.TestCase):
    def test_334(self):
        # symtensor contraction
        sym1 = ['++-', [ni,nj,nk], None, None]
        sym2 = ['++-', [nk,nl,nm], None, None]
        ijk = st.random([nbond,]*3, sym1, tb=tb)
        klm = st.random([nbond,]*3, sym2, tb=tb)
        ijlm = st.einsum('ijk,klm->ijlm',ijk,klm)
        # sparse tensor contraction
        ijk_dense, klm_dense = ijk.make_dense(), klm.make_dense()
        ijlm_dense = st.einsum('IJKijk,KLMklm->IJLMijlm', ijk_dense, klm_dense)
        ijlm_sparse = st.einsum('IJLMijlm,IJLM->IJLijlm',ijlm_dense, ijlm.get_irrep_map())
        diff = (ijlm-ijlm_sparse).norm() / numpy.sqrt(ijlm_sparse.size)
        self.assertTrue(diff<thresh)

    def test_444(self):
        numpy.random.seed(3)
        sym1 = ['+++-', [ni,nj,nl,nm], None, None]
        sym2 = ['++--', [nn,no,nj,nl], None, None]
        ijlm = st.random([nbond,]*4, sym1, tb=tb)
        nojl = st.random([nbond,]*4, sym2, tb=tb)
        inom = st.einsum('ijlm,nojl->inom', ijlm, nojl)
        equal = check_sym_equal(sym1, sym2)

        ijlm_dense, nojl_dense = ijlm.make_dense(), nojl.make_dense()
        inom_dense = st.einsum('IJLMijlm,NOJLnojl->INOMinom', ijlm_dense, nojl_dense)
        inom_sparse = st.einsum('INOMinom,INOM->INOinom', inom_dense, inom.get_irrep_map())
        diff = (inom-inom_sparse).norm() / numpy.sqrt(inom_sparse.size)
        self.assertTrue(diff<thresh)


    def test_433(self):
        sym1 = ['+++-', [ni,nn,no,nm], None, None]
        sym2 = ['++-', [ni,nn,np], None, None]
        inom = st.random([nbond,]*4, sym1, tb=tb)
        inp = st.random([nbond,]*3, sym2, tb=tb)
        pom = st.einsum('inom,inp->pom', inom, inp)

        inom_dense, inp_dense = inom.make_dense(), inp.make_dense()
        pom_dense = st.einsum('INOMinom,INPinp->POMpom', inom_dense, inp_dense)
        pom_sparse = st.einsum('POMpom,POM->POpom', pom_dense, pom.get_irrep_map())

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

        ijkl = st.random([nbond,]*4, sym1, tb=tb)
        jklmn = st.random([nbond,]*5, sym2, tb=tb)
        imn = st.einsum('ijkl,jklmn->imn', ijkl,jklmn)

        ijkl_dense, jklmn_dense = ijkl.make_dense(), jklmn.make_dense()
        imn_dense = st.einsum('IJKLijkl,JKLMNjklmn->IMNimn', ijkl_dense,jklmn_dense)
        imn_sparse = st.einsum('IMNimn,IMN->IMimn', imn_dense, imn.get_irrep_map())

        diff = (imn-imn_sparse).norm() / numpy.sqrt(imn_sparse.size)
        self.assertTrue(diff<thresh)

if __name__ == '__main__':
    unittest.main()
