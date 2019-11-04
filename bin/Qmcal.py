#! /usr/bin/env python3
from multiprocessing import Queue,Process,Manager
from TensorMol import  *
from ESOI_HDNN_MD import *
from ESOI_HDNN_MD import UpdateGPARAMS,GPARAMS 
from ESOI_HDNN_MD.Train import calculator 
import os
import math
import argparse as arg
parser=arg.ArgumentParser(description="Calculate QM reference value for MSet")
parser.add_argument('-i',"--ctrlfile")
parser.add_argument('-d',"--dataset")
args=parser.parse_args()
UpdateGPARAMS(args.ctrlfile)
TMPset=MSet(args.dataset)
TMPset.Load()
mollist=[]
for i in range(len(TMPset.mols)):
    TMPset.mols[i].properties={}
    print ("OOOOOOOOOOOOOOO")
    para=[TMPset.mols[i],'./','./',GPARAMS.Compute_setting.Gaussiankeywords,GPARAMS.Compute_setting.Ncoresperthreads,GPARAMS.Compute_setting.Atomizationlevel]
    flag,mol=calculator(para)
    if flag==True:
        mollist.append(mol)
TMPset.mols=mollist
TMPset.Save()

