from multiprocessing import Queue,Process,Manager
from TensorMol import  *
from ESOI_HDNN_MD import *
import os
import math

TMPset=MSet('Added_1AAY')
TMPset.Load()
def Get_resp_input(mollist,id):
    os.system('mkdir part%d'%id)
    for i in range(len(mollist)):
        mollist[i].name="MODEL%d"%i
        mollist[i].Write_Gaussian_input(" M062X/SDD SCF=Tight force Pop=MK",'./',14,600)
size=2000
partnum=math.floor(len(TMPset.mols)/size)
print (partnum)
plist=[]
for i in range(partnum):
    mollist=TMPset.mols[i*size:(i+1)*size]
    p=Process(target=Get_resp_input,args=(mollist,i))
    plist.append(p)
    plist[-1].start()
for i in range(partnum):
    plist[i].join()    
