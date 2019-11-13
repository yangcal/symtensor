import unittest
import numpy as np
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
    bkn = np
elif backend=='ctf':
    import ctf
    bkn = ctf

cell = gto.M(a = np.eye(3)*5,atom = '''He 0 0 0''',basis = 'gth-szv',verbose=0)
kpts = cell.make_kpts([2,2,1]) + np.random.random([1,3])
gvec = cell.reciprocal_vectors()
nkpts, nocc, nvir = len(kpts), 5, 8
kconserv = tools.get_kconserv(cell, kpts)
thresh = 1e-6

class PBCNUMPYTest(unittest.TestCase):

    def test_222(self):
        A = random('+-', [kpts,]*2, [nocc,nocc], backend, gvec, None)
        B = random('+-', [kpts,]*2, [nocc,nvir], backend, gvec, None)
        C = random('+-', [kpts,]*2, [nvir,nvir], backend, gvec, None)
        A_sparse, B_sparse, C_sparse = A.make_sparse(), B.make_sparse(), C.make_sparse()

        X = bkn.einsum('ACac,ICic->IAia', C_sparse, B_sparse)
        X = bkn.einsum('IAia,IA->Iia', X, A.get_irrep_map())

        X1 = einsum('ac,ic->ia', C, B)
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y = bkn.einsum('KIki,KAka->IAia', A_sparse, B_sparse)
        Y = bkn.einsum('IAia,IA->Iia', Y, A.get_irrep_map())
        Y1 = einsum('ki,ka->ia', A, B)
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_224(self):
        A = random('+-', [kpts,]*2, [nocc,nvir], backend, gvec, None)
        B = random('+-', [kpts,]*2, [nocc,nvir], backend, gvec, None)
        A_sparse, B_sparse = A.make_sparse(), B.make_sparse()

        X = bkn.einsum('IAia,JBjb->IJABijab', A_sparse, B_sparse)
        X = bkn.einsum('IJABijab,JB->IJAijab', X, A.get_irrep_map())

        X1 = einsum('ia,jb->ijab', A, B)
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)


    def test_242(self):
        A = random('+-', [kpts,]*2, [nocc,nvir], backend, gvec, None)
        B = random('++--', [kpts,]*4, [nocc,nocc,nvir, nvir], backend, gvec, None)
        A_sparse, B_sparse = A.make_sparse(), B.make_sparse()

        X = bkn.einsum('KCkc,KICAkica->IAia', A_sparse, B_sparse)
        X = bkn.einsum('IAia,IA->Iia', X, A.get_irrep_map())

        X1 = einsum('kc,kica->ia', A, B)
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

    def test_244(self):
        A = random('+-', [kpts,]*2, [nocc,nvir], backend, gvec, None)
        B = random('+-+-', [kpts,]*4, [nocc,nvir,nocc,nocc], backend, gvec, None)
        C = random('+-+-', [kpts,]*4, [nocc,nvir,nvir, nvir], backend, gvec, None)
        A_sparse, B_sparse, C_sparse = A.make_sparse(), B.make_sparse(), C.make_sparse()

        X1 = einsum('kclj,ic->klij', B, A)
        X = bkn.einsum('KCLJkclj,ICic->KLIJklij', B_sparse, A_sparse)
        X = bkn.einsum('KLIJklij,KLIJ->KLIklij', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('lcki,jc->klij', B, A)
        Y = bkn.einsum('LCKIlcki,JCjc->KLIJklij', B_sparse, A_sparse)
        Y = bkn.einsum('KLIJklij,KLIJ->KLIklij', Y, Y1.get_irrep_map())
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)

        Z1 = einsum('kcad,id->akic', C, A)
        Z = bkn.einsum('KCADkcad,IDid->AKICakic', C_sparse, A_sparse)
        Z = bkn.einsum('AKICakic,AKIC->AKIakic', Z, Z1.get_irrep_map())

        diff = (Z1-Z).norm() / np.sqrt(Z.size)
        self.assertTrue(diff<thresh)

    def test_442(self):
        A = random('+-+-', [kpts,]*4, [nocc,nvir,nocc,nvir], backend, gvec, None)
        B = random('++--', [kpts,]*4, [nocc,nocc,nvir,nvir], backend, gvec, None)
        C = random('+-+-', [kpts,]*4, [nocc,nvir,nvir,nvir], backend, gvec, None)
        A_sparse, B_sparse, C_sparse = A.make_sparse(), B.make_sparse(), C.make_sparse()

        X1 = einsum('kcld,ilcd->ki', A, B)
        X = bkn.einsum('KCLDkcld,ILCDilcd->KIki', A_sparse, B_sparse)
        X = bkn.einsum('KIki,KI->Kki', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('kdac,ikcd->ia', C, B)
        Y = bkn.einsum('KDACkdac,IKCDikcd->IAia', C_sparse, B_sparse)
        Y = bkn.einsum('IAia,IA->Iia', Y, Y1.get_irrep_map())

        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)

    def test_444(self):
        B = random('+-+-', [kpts,]*4, [nocc,nvir,nocc,nvir], backend, gvec, None)
        C = random('++--', [kpts,]*4, [nocc,nocc,nvir,nvir], backend, gvec, None)
        D = random('++--', [kpts,]*4, [nvir,nvir,nvir,nvir], backend, gvec, None)
        B_sparse, C_sparse, D_sparse = B.make_sparse(), C.make_sparse(), D.make_sparse()

        X1 = einsum('kcld,ijcd->klij', B, C)
        X = bkn.einsum('KCLDkcld,IJCDijcd->KLIJklij', B_sparse, C_sparse)
        X = bkn.einsum('KLIJklij,KLIJ->KLIklij', X, X1.get_irrep_map())


        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('ldkc,ilda->akic', B, C)
        Y = bkn.einsum('LDKCldkc,ILDAilda->AKICakic', B_sparse, C_sparse)
        Y = bkn.einsum('AKICakic,AKIC->AKIakic', Y, Y1.get_irrep_map())
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)

        Z1 = einsum('abcd,ijcd->ijab', D, C)
        Z = bkn.einsum('ABCDabcd,IJCDijcd->IJABijab', D_sparse, C_sparse)
        Z = bkn.einsum('IJABijab,IJAB->IJAijab', Z, Z1.get_irrep_map())

        diff = (Z1-Z).norm() / np.sqrt(Z.size)
        self.assertTrue(diff<thresh)

    def test_343(self):
        kshift=1
        klij = random('++--', [kpts,]*4, [nocc,nocc,nocc,nocc], backend, gvec, None)
        klb = random('++-', [kpts,]*3, [nocc,nocc,nvir], backend, gvec, kpts[kshift])
        lbdj = random('++--', [kpts,]*4, [nocc,nvir,nvir,nocc], backend, gvec, None)
        ild = random('++-', [kpts,]*3, [nocc,nocc,nvir], backend, gvec, kpts[kshift])
        klij_sparse, klb_sparse, lbdj_sparse, ild_sparse = klij.make_sparse(), klb.make_sparse(), lbdj.make_sparse(), ild.make_sparse()

        X1 = einsum('klij,klb->ijb', klij, klb)
        X = bkn.einsum('KLIJklij,KLBklb->IJBijb', klij_sparse, klb_sparse)
        X = bkn.einsum('IJBijb,IJB->IJijb', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('lbdj,ild->ijb', lbdj, ild)
        Y = bkn.einsum('LBDJlbdj,ILDild->IJBijb', lbdj_sparse, ild_sparse)
        Y = bkn.einsum('IJBijb,IJB->IJijb', Y, Y1.get_irrep_map())

        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)

    def test_431(self):
        kshift = 0
        lkdc = random('++--', [kpts,]*4, [nocc,nocc,nvir,nvir], backend, gvec, None)
        kld = random('++-', [kpts,]*3, [nocc,nocc,nvir], backend, gvec, kpts[kshift])
        kldc = random('++--', [kpts,]*4, [nocc,nocc,nvir,nvir], backend, gvec, None)
        lkdc_sparse, kld_sparse, kldc_sparse = lkdc.make_sparse(), kld.make_sparse(), kldc.make_sparse()

        X1 = einsum('lkdc,kld->c', lkdc, kld)
        X = bkn.einsum('LKDClkdc,KLDkld->Cc', lkdc_sparse, kld_sparse)
        X = bkn.einsum('Cc,C->c', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('kldc,kld->c', kldc, kld)
        Y = bkn.einsum('KLDCkldc,KLDkld->Cc', kldc_sparse, kld_sparse)
        Y = bkn.einsum('Cc,C->c', Y, Y1.get_irrep_map())

        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)

    def test_211(self):
        kshift= 2
        ki = random('+-', [kpts,]*2, [nocc,nocc], backend, gvec, None)
        k = random('+', [kpts], [nocc], backend, gvec, kpts[kshift])
        ki_sparse, k_sparse = ki.make_sparse(), k.make_sparse()

        X1 = einsum('ki,k->i',ki, k)
        X = bkn.einsum('KIki,Kk->Ii', ki_sparse, k_sparse)
        X = bkn.einsum('Ii,I->i', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

    def test_231(self):
        kshift= 0
        ld = random('+-', [kpts,]*2, [nocc,nvir], backend, gvec, None)
        ild = random('++-', [kpts,]*3, [nocc,nocc,nvir], backend, gvec, kpts[kshift])
        ld_sparse, ild_sparse = ld.make_sparse(), ild.make_sparse()

        X1 = einsum('ld,ild->i', ld, ild)
        X = bkn.einsum('LDld,ILDild->Ii', ld_sparse, ild_sparse)
        X = bkn.einsum('Ii,I->i', X, X1.get_irrep_map())
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

    def test_413(self):
        kshift = 1
        c = random('+', [kpts], [nvir], backend, gvec, kpts[kshift])
        ijcb = random('++--', [kpts,]*4, [nocc,nocc,nvir,nvir], backend, gvec, None)
        c_sparse, ijcb_sparse = c.make_sparse(), ijcb.make_sparse()

        X1 = einsum('c,ijcb->ijb', c, ijcb)
        X = bkn.einsum('Cc,IJCBijcb->IJBijb', c_sparse, ijcb_sparse)
        X = bkn.einsum('IJBijb,IJB->IJijb',X,X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

    def test_233(self):
        kshift = 2
        bd = random('+-', [kpts,]*2, [nvir,nvir], backend, gvec, None)
        ijd = random('++-', [kpts,]*3, [nocc,nocc,nvir], backend, gvec, kpts[kshift])
        ki = random('+-', [kpts,]*2, [nocc,nocc], backend, gvec, None)
        kjb = random('++-', [kpts,]*3, [nocc,nocc,nvir], backend, gvec, kpts[kshift])
        bd_sparse, ijd_sparse, ki_sparse, kjb_sparse = bd.make_sparse(), ijd.make_sparse(), ki.make_sparse(), kjb.make_sparse()

        X1 = einsum('bd,ijd->ijb', bd, ijd)
        X = bkn.einsum('BDbd,IJDijd->IJBijb', bd_sparse, ijd_sparse)
        X = bkn.einsum('IJBijb,IJB->IJijb', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('ki,kjb->ijb', ki, kjb)
        Y = bkn.einsum('KIki,KJBkjb->IJBijb', ki_sparse, kjb_sparse)
        Y = bkn.einsum('IJBijb,IJB->IJijb', Y, Y1.get_irrep_map())
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)

    def test_440(self):
        ijab = random('++--', [kpts,]*4, [nocc,nocc,nvir,nvir], backend, gvec, None)
        ijba = random('++--', [kpts,]*4, [nocc,nocc,nvir,nvir], backend, gvec, None)
        ijab_sparse, ijba_sparse = ijab.make_sparse(), ijba.make_sparse()

        X = bkn.einsum('IJABijab,IJBAijba->', ijab_sparse, ijba_sparse)
        X1 = einsum('ijab,ijba->', ijab, ijba)
        diff = abs(X-X1)
        self.assertTrue(diff<thresh)

if __name__ == '__main__':
    unittest.main()
