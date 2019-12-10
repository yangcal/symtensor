import unittest
import numpy as np
from symtensor.symlib import SYMLIB
from symtensor.sym import random, einsum, core_einsum
from pyscf.pbc import gto
import pyscf.pbc.tools.pbc as tools

cell = gto.M(a = np.eye(3)*5,atom = '''He 0 0 0''',basis = 'gth-szv',verbose=0)
kpts = cell.make_kpts([2,2,1]) + np.random.random([1,3])
gvec = cell.reciprocal_vectors()
nkpts, nocc, nvir = len(kpts), 5, 8
kconserv = tools.get_kconserv(cell, kpts)
thresh = 1e-6
kshift=2
sym_phys = ['++--', [kpts,]*4, None, gvec]
sym_chem = ['+-+-', [kpts,]*4, None, gvec]
sym_t1 = ['+-',[kpts,]*2, None, gvec]
sym_eom = ['++-', [kpts,]*3, kpts[kshift], gvec]
sym_s = ['+', [kpts], kpts[kshift], gvec]
symlib = SYMLIB('numpy')#.update(sym_phys, sym_chem, sym_t1, sym_eom, sym_s)

class PBCNUMPYTest(unittest.TestCase):

    def test_222(self):
        A = random([nocc,nocc],sym_t1)
        B = random([nocc,nvir],sym_t1)
        C = random([nvir,nvir],sym_t1)
        A_sparse, B_sparse, C_sparse = A.make_sparse(), B.make_sparse(), C.make_sparse()

        X = core_einsum('ACac,ICic->IAia', C_sparse, B_sparse)
        X = core_einsum('IAia,IA->Iia', X, A.get_irrep_map())

        X1 = einsum('ac,ic->ia', C, B, symlib=symlib)
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y = core_einsum('KIki,KAka->IAia', A_sparse, B_sparse)
        Y = core_einsum('IAia,IA->Iia', Y, A.get_irrep_map())
        Y1 = einsum('ki,ka->ia', A, B, symlib=symlib)
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_224(self):
        A = random([nocc,nvir],sym_t1)
        B = random([nocc,nvir],sym_t1)
        A_sparse, B_sparse = A.make_sparse(), B.make_sparse()
        X = core_einsum('IAia,JBjb->IJABijab', A_sparse, B_sparse)
        X = core_einsum('IJABijab,JB->IJAijab', X, A.get_irrep_map())

        X1 = einsum('ia,jb->ijab', A, B, symlib=symlib)
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)



    def test_242(self):
        A = random([nocc,nvir],sym_t1)
        B = random([nocc,nocc,nvir,nvir],sym_phys)
        A_sparse, B_sparse = A.make_sparse(), B.make_sparse()

        X = core_einsum('KCkc,KICAkica->IAia', A_sparse, B_sparse)
        X = core_einsum('IAia,IA->Iia', X, A.get_irrep_map())

        X1 = einsum('kc,kica->ia', A, B, symlib=symlib)
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

    def test_244(self):
        A = random([nocc,nvir], sym_t1)
        B = random([nocc,nvir,nocc,nocc], sym_chem)
        C = random([nocc,nvir,nvir,nvir], sym_chem)
        A_sparse, B_sparse, C_sparse = A.make_sparse(), B.make_sparse(), C.make_sparse()

        X1 = einsum('kclj,ic->klij', B, A, symlib=symlib)
        X = core_einsum('KCLJkclj,ICic->KLIJklij', B_sparse, A_sparse)
        X = core_einsum('KLIJklij,KLIJ->KLIklij', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('lcki,jc->klij', B, A, symlib=symlib)
        Y = core_einsum('LCKIlcki,JCjc->KLIJklij', B_sparse, A_sparse)
        Y = core_einsum('KLIJklij,KLIJ->KLIklij', Y, Y1.get_irrep_map())
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)

        Z1 = einsum('kcad,id->akic', C, A, symlib=symlib)
        Z = core_einsum('KCADkcad,IDid->AKICakic', C_sparse, A_sparse)
        Z = core_einsum('AKICakic,AKIC->AKIakic', Z, Z1.get_irrep_map())

        diff = (Z1-Z).norm() / np.sqrt(Z.size)
        self.assertTrue(diff<thresh)

    def test_442(self):
        A = random([nocc,nvir,nocc,nvir], sym_chem)
        B = random([nocc,nocc,nvir,nvir], sym_phys)
        C = random([nocc,nvir,nvir,nvir], sym_chem)
        A_sparse, B_sparse, C_sparse = A.make_sparse(), B.make_sparse(), C.make_sparse()

        X1 = einsum('kcld,ilcd->ki', A, B, symlib=symlib)
        X = core_einsum('KCLDkcld,ILCDilcd->KIki', A_sparse, B_sparse)
        X = core_einsum('KIki,KI->Kki', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('kdac,ikcd->ia', C, B, symlib=symlib)
        Y = core_einsum('KDACkdac,IKCDikcd->IAia', C_sparse, B_sparse)
        Y = core_einsum('IAia,IA->Iia', Y, Y1.get_irrep_map())

        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_444(self):
        B = random([nocc,nvir,nocc,nvir], sym_chem)
        C = random([nocc,nocc,nvir,nvir], sym_phys)
        D = random([nvir,nvir,nvir,nvir], sym_phys)
        B_sparse, C_sparse, D_sparse = B.make_sparse(), C.make_sparse(), D.make_sparse()

        X1 = einsum('kcld,ijcd->klij', B, C, symlib=symlib)
        X = core_einsum('KCLDkcld,IJCDijcd->KLIJklij', B_sparse, C_sparse)
        X = core_einsum('KLIJklij,KLIJ->KLIklij', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('ldkc,ilda->akic', B, C, symlib=symlib)
        Y = core_einsum('LDKCldkc,ILDAilda->AKICakic', B_sparse, C_sparse)
        Y = core_einsum('AKICakic,AKIC->AKIakic', Y, Y1.get_irrep_map())
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)

        Z1 = einsum('abcd,ijcd->ijab', D, C, symlib=symlib)
        Z = core_einsum('ABCDabcd,IJCDijcd->IJABijab', D_sparse, C_sparse)
        Z = core_einsum('IJABijab,IJAB->IJAijab', Z, Z1.get_irrep_map())

        diff = (Z1-Z).norm() / np.sqrt(Z.size)
        self.assertTrue(diff<thresh)


    def test_343(self):
        klij = random([nocc,nocc,nocc,nocc], sym_phys)
        klb = random([nocc,nocc,nvir], sym_eom)
        lbdj = random([nocc,nvir,nvir,nocc], sym_phys)
        ild = random([nocc,nocc,nvir], sym_eom)
        klij_sparse, klb_sparse, lbdj_sparse, ild_sparse = klij.make_sparse(), klb.make_sparse(), lbdj.make_sparse(), ild.make_sparse()

        X1 = einsum('klij,klb->ijb', klij, klb, symlib=symlib)
        X = core_einsum('KLIJklij,KLBklb->IJBijb', klij_sparse, klb_sparse)
        X = core_einsum('IJBijb,IJB->IJijb', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('lbdj,ild->ijb', lbdj, ild, symlib=symlib)
        Y = core_einsum('LBDJlbdj,ILDild->IJBijb', lbdj_sparse, ild_sparse)
        Y = core_einsum('IJBijb,IJB->IJijb', Y, Y1.get_irrep_map())

        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_431(self):

        lkdc = random([nocc,nocc,nvir,nvir], sym_phys)
        kld = random([nocc,nocc,nvir], sym_eom)
        kldc = random([nocc,nocc,nvir,nvir], sym_phys)
        lkdc_sparse, kld_sparse, kldc_sparse = lkdc.make_sparse(), kld.make_sparse(), kldc.make_sparse()

        X1 = einsum('lkdc,kld->c', lkdc, kld, symlib=symlib)
        X = core_einsum('LKDClkdc,KLDkld->Cc', lkdc_sparse, kld_sparse)
        X = core_einsum('Cc,C->c', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('kldc,kld->c', kldc, kld, symlib=symlib)
        Y = core_einsum('KLDCkldc,KLDkld->Cc', kldc_sparse, kld_sparse)
        Y = core_einsum('Cc,C->c', Y, Y1.get_irrep_map())

        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_211(self):

        ki = random([nocc,nocc], sym_t1)
        k = random([nocc], sym_s)
        ki_sparse, k_sparse = ki.make_sparse(), k.make_sparse()

        X1 = einsum('ki,k->i',ki, k, symlib=symlib)
        X = core_einsum('KIki,Kk->Ii', ki_sparse, k_sparse)
        X = core_einsum('Ii,I->i', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)


    def test_231(self):
        ld = random([nocc,nvir], sym_t1)
        ild = random([nocc,nocc,nvir], sym_eom)
        ld_sparse, ild_sparse = ld.make_sparse(), ild.make_sparse()

        X1 = einsum('ld,ild->i', ld, ild, symlib=symlib)
        X = core_einsum('LDld,ILDild->Ii', ld_sparse, ild_sparse)
        X = core_einsum('Ii,I->i', X, X1.get_irrep_map())
        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)


    def test_413(self):
        c = random([nvir], sym_s)
        ijcb = random([nocc,nocc,nvir,nvir], sym_phys)
        c_sparse, ijcb_sparse = c.make_sparse(), ijcb.make_sparse()

        X1 = einsum('c,ijcb->ijb', c, ijcb, symlib=symlib)
        X = core_einsum('Cc,IJCBijcb->IJBijb', c_sparse, ijcb_sparse)
        X = core_einsum('IJBijb,IJB->IJijb',X,X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)


    def test_233(self):
        bd = random([nvir,nvir], sym_t1)
        ijd = random([nocc,nocc,nvir], sym_eom)
        ki = random([nocc,nocc], sym_t1)
        kjb = random([nocc,nocc,nvir], sym_eom)
        bd_sparse, ijd_sparse, ki_sparse, kjb_sparse = bd.make_sparse(), ijd.make_sparse(), ki.make_sparse(), kjb.make_sparse()

        X1 = einsum('bd,ijd->ijb', bd, ijd, symlib=symlib)
        X = core_einsum('BDbd,IJDijd->IJBijb', bd_sparse, ijd_sparse)
        X = core_einsum('IJBijb,IJB->IJijb', X, X1.get_irrep_map())

        diff = (X1-X).norm() / np.sqrt(X.size)
        self.assertTrue(diff<thresh)

        Y1 = einsum('ki,kjb->ijb', ki, kjb, symlib=symlib)
        Y = core_einsum('KIki,KJBkjb->IJBijb', ki_sparse, kjb_sparse)
        Y = core_einsum('IJBijb,IJB->IJijb', Y, Y1.get_irrep_map())
        diff = (Y1-Y).norm() / np.sqrt(Y.size)
        self.assertTrue(diff<thresh)


    def test_440(self):
        ijab = random([nocc,nocc,nvir,nvir], sym_phys)
        ijba = random([nocc,nocc,nvir,nvir], sym_phys)
        ijab_sparse, ijba_sparse = ijab.make_sparse(), ijba.make_sparse()

        X = core_einsum('IJABijab,IJBAijba->', ijab_sparse, ijba_sparse)
        X1 = einsum('ijab,ijba->', ijab, ijba, symlib=symlib)
        diff = abs(X-X1)
        self.assertTrue(diff<thresh)


if __name__ == '__main__':
    unittest.main()
