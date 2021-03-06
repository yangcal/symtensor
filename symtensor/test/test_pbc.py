#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
import unittest
import numpy as np
from symtensor import random, einsum

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
        A = random.random([nocc,nocc],sym_t1)
        B = random.random([nocc,nvir],sym_t1)
        C = random.random([nvir,nvir],sym_t1)
        A_dense, B_dense, C_dense = A.make_dense(), B.make_dense(), C.make_dense()

        X = einsum('ACac,ICic->IAia', C_dense, B_dense)
        X = einsum('IAia,IA->Iia', X, A.get_irrep_map())

        X1 = einsum('ac,ic->ia', C, B)
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y = einsum('KIki,KAka->IAia', A_dense, B_dense)
        Y = einsum('IAia,IA->Iia', Y, A.get_irrep_map())
        Y1 = einsum('ki,ka->ia', A, B)
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_224(self):
        A = random.random([nocc,nvir],sym_t1)
        B = random.random([nocc,nvir],sym_t1)
        A_dense, B_dense = A.make_dense(), B.make_dense()
        X = einsum('IAia,JBjb->IJABijab', A_dense, B_dense)
        X = einsum('IJABijab,JB->IJAijab', X, A.get_irrep_map())

        X1 = einsum('ia,jb->ijab', A, B)
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)



    def test_242(self):
        A = random.random([nocc,nvir],sym_t1)
        B = random.random([nocc,nocc,nvir,nvir],sym_phys)
        A_dense, B_dense = A.make_dense(), B.make_dense()

        X = einsum('KCkc,KICAkica->IAia', A_dense, B_dense)
        X = einsum('IAia,IA->Iia', X, A.get_irrep_map())

        X1 = einsum('kc,kica->ia', A, B)
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

    def test_244(self):
        A = random.random([nocc,nvir], sym_t1)
        B = random.random([nocc,nvir,nocc,nocc], sym_chem)
        C = random.random([nocc,nvir,nvir,nvir], sym_chem)
        A_dense, B_dense, C_dense = A.make_dense(), B.make_dense(), C.make_dense()

        X1 = einsum('kclj,ic->klij', B, A)
        X = einsum('KCLJkclj,ICic->KLIJklij', B_dense, A_dense)
        X = einsum('KLIJklij,KLIJ->KLIklij', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('lcki,jc->klij', B, A)
        Y = einsum('LCKIlcki,JCjc->KLIJklij', B_dense, A_dense)
        Y = einsum('KLIJklij,KLIJ->KLIklij', Y, Y1.get_irrep_map())
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)

        Z1 = einsum('kcad,id->akic', C, A)
        Z = einsum('KCADkcad,IDid->AKICakic', C_dense, A_dense)
        Z = einsum('AKICakic,AKIC->AKIakic', Z, Z1.get_irrep_map())

        diff = (Z1-Z).norm() / np.sqrt(Z.size)
        self.assertTrue(diff<thresh)

    def test_442(self):
        A = random.random([nocc,nvir,nocc,nvir], sym_chem)
        B = random.random([nocc,nocc,nvir,nvir], sym_phys)
        C = random.random([nocc,nvir,nvir,nvir], sym_chem)
        A_dense, B_dense, C_dense = A.make_dense(), B.make_dense(), C.make_dense()

        X1 = einsum('kcld,ilcd->ki', A, B)
        X = einsum('KCLDkcld,ILCDilcd->KIki', A_dense, B_dense)
        X = einsum('KIki,KI->Kki', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('kdac,ikcd->ia', C, B)
        Y = einsum('KDACkdac,IKCDikcd->IAia', C_dense, B_dense)
        Y = einsum('IAia,IA->Iia', Y, Y1.get_irrep_map())

        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_444(self):
        B = random.random([nocc,nvir,nocc,nvir], sym_chem)
        C = random.random([nocc,nocc,nvir,nvir], sym_phys)
        D = random.random([nvir,nvir,nvir,nvir], sym_phys)
        B_dense, C_dense, D_dense = B.make_dense(), C.make_dense(), D.make_dense()
        X1 = einsum('kcld,ijcd->klij', B, C)
        X = einsum('KCLDkcld,IJCDijcd->KLIJklij', B_dense, C_dense)
        X = einsum('KLIJklij,KLIJ->KLIklij', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('ldkc,ilda->akic', B, C)
        Y = einsum('LDKCldkc,ILDAilda->AKICakic', B_dense, C_dense)
        Y = einsum('AKICakic,AKIC->AKIakic', Y, Y1.get_irrep_map())
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)

        Z1 = einsum('abcd,ijcd->ijab', D, C)
        Z = einsum('ABCDabcd,IJCDijcd->IJABijab', D_dense, C_dense)
        Z = einsum('IJABijab,IJAB->IJAijab', Z, Z1.get_irrep_map())

        diff = (Z1-Z).norm() / np.sqrt(Z.size)
        self.assertTrue(diff<thresh)

    def test_343(self):
        klij = random.random([nocc,nocc,nocc,nocc], sym_phys)
        klb = random.random([nocc,nocc,nvir], sym_eom)
        lbdj = random.random([nocc,nvir,nvir,nocc], sym_phys)
        ild = random.random([nocc,nocc,nvir], sym_eom)
        klij_dense, klb_dense, lbdj_dense, ild_dense = klij.make_dense(), klb.make_dense(), lbdj.make_dense(), ild.make_dense()

        X1 = einsum('klij,klb->ijb', klij, klb)
        X = einsum('KLIJklij,KLBklb->IJBijb', klij_dense, klb_dense)
        X = einsum('IJBijb,IJB->IJijb', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('lbdj,ild->ijb', lbdj, ild)
        Y = einsum('LBDJlbdj,ILDild->IJBijb', lbdj_dense, ild_dense)
        Y = einsum('IJBijb,IJB->IJijb', Y, Y1.get_irrep_map())

        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_431(self):

        lkdc = random.random([nocc,nocc,nvir,nvir], sym_phys)
        kld = random.random([nocc,nocc,nvir], sym_eom)
        kldc = random.random([nocc,nocc,nvir,nvir], sym_phys)
        lkdc_dense, kld_dense, kldc_dense = lkdc.make_dense(), kld.make_dense(), kldc.make_dense()

        X1 = einsum('lkdc,kld->c', lkdc, kld)
        X = einsum('LKDClkdc,KLDkld->Cc', lkdc_dense, kld_dense)
        X = einsum('Cc,C->c', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('kldc,kld->c', kldc, kld)
        Y = einsum('KLDCkldc,KLDkld->Cc', kldc_dense, kld_dense)
        Y = einsum('Cc,C->c', Y, Y1.get_irrep_map())

        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_211(self):

        ki = random.random([nocc,nocc], sym_t1)
        k = random.random([nocc], sym_s)
        ki_dense, k_dense = ki.make_dense(), k.make_dense()

        X1 = einsum('ki,k->i',ki, k)
        X = einsum('KIki,Kk->Ii', ki_dense, k_dense)
        X = einsum('Ii,I->i', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)


    def test_231(self):
        ld = random.random([nocc,nvir], sym_t1)
        ild = random.random([nocc,nocc,nvir], sym_eom)
        ld_dense, ild_dense = ld.make_dense(), ild.make_dense()

        X1 = einsum('ld,ild->i', ld, ild)
        X = einsum('LDld,ILDild->Ii', ld_dense, ild_dense)
        X = einsum('Ii,I->i', X, X1.get_irrep_map())
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)


    def test_413(self):
        c = random.random([nvir], sym_s)
        ijcb = random.random([nocc,nocc,nvir,nvir], sym_phys)
        c_dense, ijcb_dense = c.make_dense(), ijcb.make_dense()

        X1 = einsum('c,ijcb->ijb', c, ijcb)
        X = einsum('Cc,IJCBijcb->IJBijb', c_dense, ijcb_dense)
        X = einsum('IJBijb,IJB->IJijb',X,X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)


    def test_233(self):
        bd = random.random([nvir,nvir], sym_t1)
        ijd = random.random([nocc,nocc,nvir], sym_eom)
        ki = random.random([nocc,nocc], sym_t1)
        kjb = random.random([nocc,nocc,nvir], sym_eom)
        bd_dense, ijd_dense, ki_dense, kjb_dense = bd.make_dense(), ijd.make_dense(), ki.make_dense(), kjb.make_dense()

        X1 = einsum('bd,ijd->ijb', bd, ijd)
        X = einsum('BDbd,IJDijd->IJBijb', bd_dense, ijd_dense)
        X = einsum('IJBijb,IJB->IJijb', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('ki,kjb->ijb', ki, kjb)
        Y = einsum('KIki,KJBkjb->IJBijb', ki_dense, kjb_dense)
        Y = einsum('IJBijb,IJB->IJijb', Y, Y1.get_irrep_map())
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_440(self):
        ijab = random.random([nocc,nocc,nvir,nvir], sym_phys)
        ijba = random.random([nocc,nocc,nvir,nvir], sym_phys)
        ijab_dense, ijba_dense = ijab.make_dense(), ijba.make_dense()

        X = einsum('IJABijab,IJBAijba->', ijab_dense, ijba_dense)
        X1 = einsum('ijab,ijba->', ijab, ijba)
        diff = abs(X-X1)
        self.assertTrue(diff<thresh)

if __name__ == '__main__':
    unittest.main()
