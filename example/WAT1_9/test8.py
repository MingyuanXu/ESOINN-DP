from __future__ import absolute_import
from __future__ import print_function
import sys
sys.path.append('../Esoinn')

from  TensorMol import *
from NN import *
from NN_charge import *
import pickle
import random
from global_var import *
import numpy as np
from update_mol import Molnew

TMMSET1=MSet("2l30_m062x")
TMMSET1.Load()
TMMSET=MSet("Total")

for i in range(len(TMMSET1.mols)):
    flag=True
    mol=TMMSET1.mols[i]
    natom=len(mol.atoms)
    crd=mol.coords
    maxdis=0;mindis=10
    for j in range(0,natom-1):
        for k in range(j+1,natom):
            dis=np.sqrt(np.sum((crd[j]-crd[k])**2))
            if dis>maxdis:
                maxdis=dis
            if dis<mindis:
                mindis=dis
    if mindis<0.85 or maxdis >15:
        flag=False
    force=mol.properties['force']
    max_force=np.max(np.abs(force))
    if max_force*627.51>300:
        flag=False
    if flag==True:
       TMMSET.mols.append(mol)
    else:
       print (mol,mindis,maxdis,max_force)

TMMSET.Save()

