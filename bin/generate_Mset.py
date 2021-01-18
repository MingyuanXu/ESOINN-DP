from TensorMol import  *
from ESOI_HDNN_MD import *
from ESOI_HDNN_MD.Base import *
from ESOI_HDNN_MD.Comparm import GPARAMS
from ESOI_HDNN_MD import UpdateGPARAMS,LoadModel,Added_MSet

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
            if keyword in filename:
                filelist.append([home,filename,os.path.join(home,filename)])
    return filelist

def get_mol(para):
    logpath,filename,logfile=para
    molname=filename.strip('.log')
    mol=Molnew(name=molname)
    flag1=mol.Update_from_Gaulog(logpath+'/')
    mol.CalculateAtomization(GPARAMS.Compute_setting.Atomizationlevel)
    if GPARAMS.Esoinn_setting.Ifresp==True:
        flag2,respc=cal_resp_charge(logfile.strip('_ef.log')+'_q.log')
    elif GPARAMS.Esoinn_setting.Ifadch==True:
        chgfile=open('.'+logfile.strip('_ef.log')+'_ef.chg','r')
        adchcharge=[]
        for i in range(len(mol.atoms)):
            line=chgfile.readline()
            var=line.split()
            adchcharge.append(float(var[4])) 
        adchcharge=np.array(adchcharge)
        flag2=True
    else:
        flag2=True
    flag=flag1 and flag2
    if flag==True: 
        if GPARAMS.Esoinn_setting.Ifresp==True:
            mol.properties['resp_charge']=respc
        if GPARAMS.Esoinn_setting.Ifadch==True:
            mol.properties['adch_charge']=adchcharge
            #mol.totalcharge=int(round(np.sum(adchcharge)))
        return [flag,mol]
    else:
        return [flag,None]

pool=multiprocessing.Pool(processes=4)
mollist=[]
loglist=[]
for path in pathlist:
    loglist+=get_filelist(path,'.log')
print (loglist)

mollist=pool.imap(get_mol,loglist)
for result in mollist:
    if result[0]==True:
        TMPset.mols.append (result[1])
for i in TMPset.mols:
    if GPARAMS.Esoinn_setting.Ifresp==True:
        print (i,i.properties['atomization'],len(i.properties['resp_charge']))
    else:
        print (i.name,i.properties['atomization'],len(i.properties['adch_charge']),np.sum(i.properties['adch_charge']),i.totalcharge)
TMPset.Save()

