from multiprocessing import Queue,Process,Manager
from TensorMol import  *
from ESOI_HDNN_MD import *
from ESOI_HDNN_MD.Base import *
import os
import math

TMPSet=MSet("PM6")
file=open("filelist",'r')
for eachline in file:
    mol=Molnew()
    mol.name=''
    flag=mol.Update_from_Gaulog(eachline.strip())
    if flag==True:
        mol.CalculateAtomization("PM6")
        TMPSet.mols.append(mol)
TMPSet.Save()
print (TMPSet.mols[0].properties)
