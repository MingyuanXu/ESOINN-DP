#!/usr/bin/env python
# coding=utf-8
from TensorMol import *
from ..Comparm import GPARAMS

def EvalSet(mol_set, \
            instance,\
            Rr_cut=GPARAMS.Neuralnetwork_setting.AN1_r_Rc,\
            Ra_cut=GPARAMS.Neuralnetwork_setting.AN1_a_Rc,\
            Ree_cut=GPARAMS.Neuralnetwork_setting.EEcutoffoff,\
            HasVdw = True):
    """
    The energy, force and dipole routine for BPs_EE. Evaluate a Set
    """
    nmols = len(mol_set.mols)
    dummy_energy = np.zeros((nmols))
    dummy_dipole = np.zeros((nmols, 3))
    #self.TData.MaxNAtoms = mol_set.MaxNAtoms()
    xyzs = np.zeros((nmols, instance.MaxNAtoms, 3), dtype = np.float64)
    dummy_grads = np.zeros((nmols, instance.MaxNAtoms, 3), dtype = np.float64)
    Zs = np.zeros((nmols, instance.MaxNAtoms), dtype = np.int32)
    natom = np.zeros((nmols), dtype = np.int32)
    for i, mol in enumerate(mol_set.mols):
        xyzs[i][:mol.NAtoms()] = mol.coords
        Zs[i][:mol.NAtoms()] = mol.atoms
        natom[i] = mol.NAtoms()
    NL = NeighborListSet(xyzs, natom, True, True, Zs, sort_=True)
    rad_p_ele, ang_t_elep, mil_jk, jk_max = NL.buildPairsAndTriplesWithEleIndex(Rr_cut, Ra_cut, instance.eles_np, instance.eles_pairs_np)
    NLEE = NeighborListSet(xyzs, natom, False, False,  None)
    rad_eep = NLEE.buildPairs(Ree_cut)
    if not HasVdw:
        Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient  = instance.evaluate([xyzs, Zs, dummy_energy, dummy_dipole, dummy_grads, rad_p_ele, ang_t_elep, rad_eep, mil_jk, 1.0/natom])
        return Etotal, Ebp, Ecc, mol_dipole, atom_charge, -JOULEPERHARTREE*gradient[0]
    else:
        Etotal, Ebp, Ebp_atom, Ecc, Evdw,  mol_dipole, atom_charge, gradient = instance.evaluate([xyzs, Zs, dummy_energy, dummy_dipole, dummy_grads, rad_p_ele, ang_t_elep, rad_eep, mil_jk, 1.0/natom])
        return Etotal, Ebp, Ebp_atom ,Ecc, Evdw, mol_dipole, atom_charge, -JOULEPERHARTREE*gradient[0]

def EvalSet_charge(mol_set, instance,Rr_cut=GPARAMS.Neuralnetwork_setting.AN1_r_Rc, Ra_cut=GPARAMS.Neuralnetwork_setting.AN1_a_Rc, Ree_cut=GPARAMS.Neuralnetwork_setting.EEcutoffoff, HasVdw = True):
    nmols = len(mol_set.mols)
    dummy_energy = np.zeros((nmols))
    dummy_dipole = np.zeros((nmols, 3))
    #self.TData.MaxNAtoms = mol_set.MaxNAtoms()
    xyzs = np.zeros((nmols, instance.MaxNAtoms, 3), dtype = np.float64)
    dummy_grads  = np.zeros((nmols, instance.MaxNAtoms, 3), dtype = np.float64)
    dummy_charge = np.zeros((nmols, instance.MaxNAtoms),dtype = np.float64)
    qtlabels = np.zeros(nmols, dtype = np.float64)
    masks        = np.zeros((nmols,instance.MaxNAtoms),dtype = np.float64)
    Zs = np.zeros((nmols, instance.MaxNAtoms), dtype = np.int32)
    natom = np.zeros((nmols), dtype = np.int32)
     
    for i, mol in enumerate(mol_set.mols):
        xyzs[i][:mol.NAtoms()] = mol.coords
        Zs[i][:mol.NAtoms()]   = mol.atoms
        natom[i]               = mol.NAtoms()
        masks[i][:mol.NAtoms()]               = np.ones( mol.NAtoms(),dtype=np.float64 )
        qtlabels[i]            = mol.properties['clabel']
    
    NL = NeighborListSet(xyzs, natom, True, True, Zs, sort_=True)
    rad_p_ele, ang_t_elep, mil_jk, jk_max = NL.buildPairsAndTriplesWithEleIndex(Rr_cut, Ra_cut, instance.eles_np, instance.eles_pairs_np)
    NLEE = NeighborListSet(xyzs, natom, False, False,  None)
    rad_eep = NLEE.buildPairs(Ree_cut)

    if not HasVdw:
        Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient  =\
             instance.evaluate([xyzs, Zs, dummy_energy, dummy_dipole, dummy_charge, qtlabels, masks, dummy_grads, rad_p_ele, ang_t_elep, rad_eep, mil_jk, 1.0/natom])
        return Etotal, Ebp, Ecc, mol_dipole, atom_charge, -JOULEPERHARTREE*gradient[0]
    else:
        Etotal, Ebp, Ebp_atom, Ecc, Evdw,  mol_dipole, atom_charge, gradient =\
             instance.evaluate([xyzs, Zs, dummy_energy, dummy_dipole, dummy_charge, qtlabels, masks, dummy_grads, rad_p_ele, ang_t_elep, rad_eep, mil_jk, 1.0/natom])
        return Etotal, Ebp, Ebp_atom ,Ecc, Evdw, mol_dipole, atom_charge, -JOULEPERHARTREE*gradient[0]

def Eval_charge(mol_set,instance, Rr_cut=GPARAMS.Neuralnetwork_setting.AN1_r_Rc, Ra_cut=GPARAMS.Neuralnetwork_setting.AN1_a_Rc, Ree_cut=GPARAMS.Neuralnetwork_setting.EEcutoffoff, HasVdw = True):
    nmols = len(mol_set.mols)
    dummy_energy = np.zeros((nmols))
    dummy_dipole = np.zeros((nmols, 3))
    
    #self.TData.MaxNAtoms = mol_set.MaxNAtoms()
    xyzs = np.zeros((nmols, instance.MaxNAtoms, 3), dtype = np.float64)
    dummy_grads  = np.zeros((nmols, instance.MaxNAtoms, 3), dtype = np.float64)
    dummy_charge = np.zeros((nmols, instance.MaxNAtoms),dtype = np.float64)
    qtlabels = np.zeros(nmols, dtype = np.float64)
    masks        = np.zeros((nmols,instance.MaxNAtoms),dtype = np.float64)
    Zs = np.zeros((nmols, instance.MaxNAtoms), dtype = np.int32)
    natom = np.zeros((nmols), dtype = np.int32)
     
    for i, mol in enumerate(mol_set.mols):
        xyzs[i][:mol.NAtoms()] = mol.coords
        Zs[i][:mol.NAtoms()]   = mol.atoms
        natom[i]               = mol.NAtoms()
        masks[i][:mol.NAtoms()]               = np.ones( mol.NAtoms(),dtype=np.float64 )
        qtlabels[i]            = mol.properties['clabel']
     
    NL = NeighborListSet(xyzs, natom, True, True, Zs, sort_=True)
    rad_p_ele, ang_t_elep, mil_jk, jk_max = NL.buildPairsAndTriplesWithEleIndex(Rr_cut, Ra_cut, instance.eles_np, instance.eles_pairs_np)
    NLEE = NeighborListSet(xyzs, natom, False, False,  None)
    rad_eep = NLEE.buildPairs(Ree_cut)
    
    if not HasVdw:
        Etotal, Ebp, Ecc, mol_dipole, atom_charge, gradient  =\
             instance.evaluate([xyzs, Zs, dummy_energy, dummy_dipole, dummy_charge, qtlabels, masks, dummy_grads, rad_p_ele, ang_t_elep, rad_eep, mil_jk, 1.0/natom])
        return atom_charge
    else:
        Etotal, Ebp, Ebp_atom, Ecc, Evdw,  mol_dipole, atom_charge, gradient =\
             instance.evaluate([xyzs, Zs, dummy_energy, dummy_dipole, dummy_charge, qtlabels, masks, dummy_grads, rad_p_ele, ang_t_elep, rad_eep, mil_jk, 1.0/natom])
        return atom_charge
         
