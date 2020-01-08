from TensorMol import  *
from ESOI_HDNN_MD import *
from ESOI_HDNN_MD.Base import *
import os
import math
import multiprocessing 
import argparse as arg
parser=arg.ArgumentParser(description="Calculate QM reference value for MSet")
parser.add_argument('-i',"--ctrlfile")
parser.add_argument('-d',"--dataset")
args=parser.parse_args()
UpdateGPARAMS(args.ctrlfile)
TMPset=MSet(args.dataset)
pathlist=['./']

def get_filelist(path,keyword):
    filelist=[]
    for home,dirs,files in os.walk(path):
        for filename in files:
            if keyword in filename and "MODEL" in filename:
                filelist.append([home,filename,os.path.join(home,filename)])
    return filelist

def get_mol(para):
    logpath,filename,logfile=para
    if GPARAMS.Esoinn_setting.Ifresp==True:
        molname=filename.strip('_ef.log')
    else:
        molname=filename.strip('.log')
    mol=Molnew(name=molname)
    flag1=mol.Update_from_Gaulog(logpath+'/')
    mol.CalculateAtomization(GPARAMS.Compute_setting.Atomizationlevel)
    if GPARAMS.Esoinn_setting.Ifresp==True:
        flag2,respc=cal_resp_charge(logfile.strip('_ef.log')+'_q.log')
    else:
        flag2=True
    flag=flag1 and flag2
    if flag==True 
        if GPARAMS.Esoinn_setting.Ifresp==True:
            mol.properties['resp_charge']=respc
        return [flag,mol]
    else:
        return [flag,None]

pool=multiprocessing.Pool(processes=32)
mollist=[]
loglist=[]
for path in pathlist:
    loglist+=get_filelist(path,'.log')

mollist=pool.imap(get_resp_mol,loglist)
for result in mollist:
    if result[0]==True:
        TMPset.mols.append (result[1])

for i in TMPset.mols:
    if GPARAMS.Esoinn_setting.Ifresp==True:
        print (i,i.properties['atomization'],len(i.properties['resp_charge']))
    else:
        print (i,i.properties['atomization'])
TMPset.Save()

