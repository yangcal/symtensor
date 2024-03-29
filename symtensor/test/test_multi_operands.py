#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

import unittest
import numpy
import symtensor as st

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


class MULTIOPERANDSTEST(unittest.TestCase):

    def test_dmrg(self):
        sym1 = ['++-', [ni,nj,nk], None, None]
        sym2 = ['++-', [nk,nl,nm], None, None]
        sym3 = ['++--', [nn,no,nj,nl], None, None]
        sym4 = ['++-', [ni,nn,np], None, None]


        ijk = st.random([nbond,]*3, sym1)
        klm = st.random([nbond,]*3, sym2)
        nojl = st.random([nbond,]*4, sym3)
        inp = st.random([nbond,]*3, sym4)

        pom = st.einsum('ijk,klm,nojl,inp->pom', ijk, klm, nojl, inp)

        ijlm = st.einsum('ijk,klm->ijlm',ijk,klm)
        inom = st.einsum('ijlm,nojl->inom', ijlm, nojl)
        pom1 = st.einsum('inom,inp->pom', inom, inp)
        self.assertTrue((pom-pom1).norm()<thresh)

    def test_cc(self):
        sym1 = ["+-", [[-1,1]]*2, None, None]
        sym2 = ["++--", [[-1,1]]*4, None, None]

        ie = st.random([nbond]*2, sym1)
        ijab = st.random([nbond]*4, sym2)
        abcd = st.random([nbond]*4, sym2)

        ejcd = st.einsum('ie,ijab,abcd->ejcd', ie, ijab, abcd)

        ejab = st.einsum('ie,ijab->ejab', ie, ijab)
        ejcd1 = st.einsum('ejab,abcd->ejcd', ejab, abcd)

        ijcd = st.einsum('ijab,abcd->ijcd', ijab, abcd)
        ejcd2= st.einsum('ie,ijcd->ejcd', ie, ijcd)

        self.assertTrue((ejcd-ejcd1).norm()<thresh)
        self.assertTrue((ejcd-ejcd2).norm()<thresh)






if __name__ == '__main__':
    unittest.main()
