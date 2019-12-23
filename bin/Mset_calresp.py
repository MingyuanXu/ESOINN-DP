from multiprocessing import Queue,Process,Manager
from TensorMol import  *
from ESOI_HDNN_MD import *
import os
import math
TMPset=MSet('M062X-SDD')
TMPset.Load()
def Get_resp_input(mollist,id):
    os.system('mkdir part%d'%id)
    for i in range(len(mollist)):
        mollist[i].name="MODEL%d"%i
        mollist[i].spin=1
        mollist[i].totalcharge=int(round(np.sum(mollist[i].properties["charge"])))
        print (len(mollist[i].atoms),mollist[i].totalcharge)
        mollist[i].Write_Gaussian_input(" HF/6-31g* SCF=Tight force Pop=MK IOp(6/33=2,6/41=10,6/42=17)",'./part%d/'%id,14,600)
size=500
partnum=math.floor(len(TMPset.mols)/size)
print (partnum)
plist=[]
for i in range(partnum+1):
    mollist=TMPset.mols[i*size:(i+1)*size]
    p=Process(target=Get_resp_input,args=(mollist,i))
    plist.append(p)
    plist[-1].start()
for i in range(partnum):
    plist[i].join()    
print (len(TMPset.mols))
