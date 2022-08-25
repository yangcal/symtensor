#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
import unittest
import numpy as np
import symtensor as st

def make_kpts(lattice, nmp):
    ks_each_axis = []
    for n in nmp:
        ks = np.arange(n, dtype=float) / n
        ks_each_axis.append(ks)
    arrays = [np.asarray(x) for x in ks_each_axis]
    nd = len(arrays)
    dims = [nd] + [len(x) for x in arrays]
    out = np.ndarray(dims)
    shape = [-1] + [1] * nd
    for i, arr in enumerate(arrays):
        out[i] = arr.reshape(shape[:nd-i])
    scaled_kpts = out.reshape(nd,-1).T
    gvec = get_reciprocal_vectors(lattice)
    kpts = np.dot(scaled_kpts, gvec)
    return kpts

def get_reciprocal_vectors(lattice):
    b = np.linalg.inv(lattice.T)
    return 2*np.pi * b

def get_kconserv(lattice, kpts):
    nkpts = kpts.shape[0]
    a = lattice / (2*np.pi)
    kconserv = np.zeros((nkpts,nkpts,nkpts), dtype=int)
    kvKLM = kpts[:,None,None,:] - kpts[:,None,:] + kpts
    for N, kvN in enumerate(kpts):
        kvKLMN = np.einsum('wx,klmx->wklm', a, kvKLM - kvN, optimize=True)
        kvKLMN_int = np.rint(kvKLMN)
        mask = np.einsum('wklm->klm', abs(kvKLMN - kvKLMN_int)) < 1e-9
        kconserv[mask] = N
    return kconserv

lattice = np.eye(3)*5
kpts = make_kpts(lattice, [2,2,1])
gvec = get_reciprocal_vectors(lattice)
nkpts, nocc, nvir = len(kpts), 3, 5
kconserv = get_kconserv(lattice, kpts)
thresh = 1e-6
kshift=2
sym_phys = ['++--', [kpts,]*4, None, gvec]
sym_chem = ['+-+-', [kpts,]*4, None, gvec]
sym_t1 = ['+-',[kpts,]*2, None, gvec]
sym_eom = ['++-', [kpts,]*3, kpts[kshift], gvec]
sym_s = ['+', [kpts], kpts[kshift], gvec]

class PBCNUMPYTest(unittest.TestCase):

    def test_222(self):
        A = st.random([nocc,nocc],sym_t1)
        B = st.random([nocc,nvir],sym_t1)
        C = st.random([nvir,nvir],sym_t1)
        A_dense, B_dense, C_dense = A.make_dense(), B.make_dense(), C.make_dense()

        X = st.einsum('ACac,ICic->IAia', C_dense, B_dense)
        X = st.einsum('IAia,IA->Iia', X, A.get_irrep_map())

        X1 = st.einsum('ac,ic->ia', C, B)
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y = st.einsum('KIki,KAka->IAia', A_dense, B_dense)
        Y = st.einsum('IAia,IA->Iia', Y, A.get_irrep_map())
        Y1 = st.einsum('ki,ka->ia', A, B)
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_224(self):
        A = st.random([nocc,nvir],sym_t1)
        B = st.random([nocc,nvir],sym_t1)
        A_dense, B_dense = A.make_dense(), B.make_dense()
        X = st.einsum('IAia,JBjb->IJABijab', A_dense, B_dense)
        X = st.einsum('IJABijab,JB->IJAijab', X, A.get_irrep_map())

        X1 = st.einsum('ia,jb->ijab', A, B)
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)



    def test_242(self):
        A = st.random([nocc,nvir],sym_t1)
        B = st.random([nocc,nocc,nvir,nvir],sym_phys)
        A_dense, B_dense = A.make_dense(), B.make_dense()

        X = st.einsum('KCkc,KICAkica->IAia', A_dense, B_dense)
        X = st.einsum('IAia,IA->Iia', X, A.get_irrep_map())

        X1 = st.einsum('kc,kica->ia', A, B)
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

    def test_244(self):
        A = st.random([nocc,nvir], sym_t1)
        B = st.random([nocc,nvir,nocc,nocc], sym_chem)
        C = st.random([nocc,nvir,nvir,nvir], sym_chem)
        A_dense, B_dense, C_dense = A.make_dense(), B.make_dense(), C.make_dense()

        X1 = st.einsum('kclj,ic->klij', B, A)
        X = st.einsum('KCLJkclj,ICic->KLIJklij', B_dense, A_dense)
        X = st.einsum('KLIJklij,KLIJ->KLIklij', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = st.einsum('lcki,jc->klij', B, A)
        Y = st.einsum('LCKIlcki,JCjc->KLIJklij', B_dense, A_dense)
        Y = st.einsum('KLIJklij,KLIJ->KLIklij', Y, Y1.get_irrep_map())
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)

        Z1 = st.einsum('kcad,id->akic', C, A)
        Z = st.einsum('KCADkcad,IDid->AKICakic', C_dense, A_dense)
        Z = st.einsum('AKICakic,AKIC->AKIakic', Z, Z1.get_irrep_map())

        diff = (Z1-Z).norm() / np.sqrt(Z.size)
        self.assertTrue(diff<thresh)

    def test_442(self):
        A = st.random([nocc,nvir,nocc,nvir], sym_chem)
        B = st.random([nocc,nocc,nvir,nvir], sym_phys)
        C = st.random([nocc,nvir,nvir,nvir], sym_chem)
        A_dense, B_dense, C_dense = A.make_dense(), B.make_dense(), C.make_dense()

        X1 = st.einsum('kcld,ilcd->ki', A, B)
        X = st.einsum('KCLDkcld,ILCDilcd->KIki', A_dense, B_dense)
        X = st.einsum('KIki,KI->Kki', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = st.einsum('kdac,ikcd->ia', C, B)
        Y = st.einsum('KDACkdac,IKCDikcd->IAia', C_dense, B_dense)
        Y = st.einsum('IAia,IA->Iia', Y, Y1.get_irrep_map())

        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_444(self):
        B = st.random([nocc,nvir,nocc,nvir], sym_chem)
        C = st.random([nocc,nocc,nvir,nvir], sym_phys)
        D = st.random([nvir,nvir,nvir,nvir], sym_phys)
        B_dense, C_dense, D_dense = B.make_dense(), C.make_dense(), D.make_dense()
        X1 = st.einsum('kcld,ijcd->klij', B, C)
        X = st.einsum('KCLDkcld,IJCDijcd->KLIJklij', B_dense, C_dense)
        X = st.einsum('KLIJklij,KLIJ->KLIklij', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = st.einsum('ldkc,ilda->akic', B, C)
        Y = st.einsum('LDKCldkc,ILDAilda->AKICakic', B_dense, C_dense)
        Y = st.einsum('AKICakic,AKIC->AKIakic', Y, Y1.get_irrep_map())
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)

        Z1 = st.einsum('abcd,ijcd->ijab', D, C)
        Z = st.einsum('ABCDabcd,IJCDijcd->IJABijab', D_dense, C_dense)
        Z = st.einsum('IJABijab,IJAB->IJAijab', Z, Z1.get_irrep_map())

        diff = (Z1-Z).norm() / np.sqrt(Z.size)
        self.assertTrue(diff<thresh)

    def test_343(self):
        klij = st.random([nocc,nocc,nocc,nocc], sym_phys)
        klb = st.random([nocc,nocc,nvir], sym_eom)
        lbdj = st.random([nocc,nvir,nvir,nocc], sym_phys)
        ild = st.random([nocc,nocc,nvir], sym_eom)
        klij_dense, klb_dense, lbdj_dense, ild_dense = klij.make_dense(), klb.make_dense(), lbdj.make_dense(), ild.make_dense()

        X1 = st.einsum('klij,klb->ijb', klij, klb)
        X = st.einsum('KLIJklij,KLBklb->IJBijb', klij_dense, klb_dense)
        X = st.einsum('IJBijb,IJB->IJijb', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = st.einsum('lbdj,ild->ijb', lbdj, ild)
        Y = st.einsum('LBDJlbdj,ILDild->IJBijb', lbdj_dense, ild_dense)
        Y = st.einsum('IJBijb,IJB->IJijb', Y, Y1.get_irrep_map())

        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_431(self):

        lkdc = st.random([nocc,nocc,nvir,nvir], sym_phys)
        kld = st.random([nocc,nocc,nvir], sym_eom)
        kldc = st.random([nocc,nocc,nvir,nvir], sym_phys)
        lkdc_dense, kld_dense, kldc_dense = lkdc.make_dense(), kld.make_dense(), kldc.make_dense()

        X1 = st.einsum('lkdc,kld->c', lkdc, kld)
        X = st.einsum('LKDClkdc,KLDkld->Cc', lkdc_dense, kld_dense)
        X = st.einsum('Cc,C->c', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = st.einsum('kldc,kld->c', kldc, kld)
        Y = st.einsum('KLDCkldc,KLDkld->Cc', kldc_dense, kld_dense)
        Y = st.einsum('Cc,C->c', Y, Y1.get_irrep_map())

        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_211(self):

        ki = st.random([nocc,nocc], sym_t1)
        k = st.random([nocc], sym_s)
        ki_dense, k_dense = ki.make_dense(), k.make_dense()

        X1 = st.einsum('ki,k->i',ki, k)
        X = st.einsum('KIki,Kk->Ii', ki_dense, k_dense)
        X = st.einsum('Ii,I->i', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)


    def test_231(self):
        ld = st.random([nocc,nvir], sym_t1)
        ild = st.random([nocc,nocc,nvir], sym_eom)
        ld_dense, ild_dense = ld.make_dense(), ild.make_dense()

        X1 = st.einsum('ld,ild->i', ld, ild)
        X = st.einsum('LDld,ILDild->Ii', ld_dense, ild_dense)
        X = st.einsum('Ii,I->i', X, X1.get_irrep_map())
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)


    def test_413(self):
        c = st.random([nvir], sym_s)
        ijcb = st.random([nocc,nocc,nvir,nvir], sym_phys)
        c_dense, ijcb_dense = c.make_dense(), ijcb.make_dense()

        X1 = st.einsum('c,ijcb->ijb', c, ijcb)
        X = st.einsum('Cc,IJCBijcb->IJBijb', c_dense, ijcb_dense)
        X = st.einsum('IJBijb,IJB->IJijb',X,X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)


    def test_233(self):
        bd = st.random([nvir,nvir], sym_t1)
        ijd = st.random([nocc,nocc,nvir], sym_eom)
        ki = st.random([nocc,nocc], sym_t1)
        kjb = st.random([nocc,nocc,nvir], sym_eom)
        bd_dense, ijd_dense, ki_dense, kjb_dense = bd.make_dense(), ijd.make_dense(), ki.make_dense(), kjb.make_dense()

        X1 = st.einsum('bd,ijd->ijb', bd, ijd)
        X = st.einsum('BDbd,IJDijd->IJBijb', bd_dense, ijd_dense)
        X = st.einsum('IJBijb,IJB->IJijb', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = st.einsum('ki,kjb->ijb', ki, kjb)
        Y = st.einsum('KIki,KJBkjb->IJBijb', ki_dense, kjb_dense)
        Y = st.einsum('IJBijb,IJB->IJijb', Y, Y1.get_irrep_map())
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_440(self):
        ijab = st.random([nocc,nocc,nvir,nvir], sym_phys)
        ijba = st.random([nocc,nocc,nvir,nvir], sym_phys)
        ijab_dense, ijba_dense = ijab.make_dense(), ijba.make_dense()

        X = st.einsum('IJABijab,IJBAijba->', ijab_dense, ijba_dense)
        X1 = st.einsum('ijab,ijba->', ijab, ijba)
        diff = abs(X-X1)
        self.assertTrue(diff<thresh)

if __name__ == '__main__':
    unittest.main()
