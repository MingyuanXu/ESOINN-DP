from TensorMol import  *
from ESOI_HDNN_MD import *
from ESOI_HDNN_MD.Base import *
import os
import math
import multiprocessing 
TMPset=MSet('2l30_resp')
pathlist=['./']
def get_filelist(path,keyword):
    filelist=[]
    for home,dirs,files in os.walk(path):
        for filename in files:
            if keyword in filename:
                filelist.append([home,filename,os.path.join(home,filename)])
    return filelist 
def get_resp_mol(para):
    logpath,filename,logfile=para
    mol=Molnew(name=filename.strip('.log'))
    mol.Update_from_Gaulog(logpath+'/')
    mol.CalculateAtomization('HF/6-31g*')
    flag,respc=cal_resp_charge(logfile)
    if flag==True:
        mol.properties['resp_charge']=respc
        return [flag,mol]
    else:
        print ("RESP of %s is failed!"%mol.name)
        return [flag,None]
pool=multiprocessing.Pool(processes=4)

loglist=[]
for path in pathlist:
    loglist+=get_filelist(path,'.log')
mollist=pool.imap(get_resp_mol,loglist[:8])
#print (mollist)
for result in mollist:
    if result[0]==True:
        TMPset.mols.append (result[1])
TMPset.Save()
        
