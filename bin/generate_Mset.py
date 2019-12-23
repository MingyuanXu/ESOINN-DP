from multiprocessing import Queue,Process,Manager
from TensorMol import  *
from ESOI_HDNN_MD import *
from ESOI_HDNN_MD.Base import *
import os
import math
TMPset=MSet('2l30_resp')
pathlist=['./']
def get_filelist(path,keyword):
    filelist=[]
    for home,dirs,files in os.walk(path):
        for filename in files:
            if keyword in filename:
                filelist.append([home,filename,os.path.join(home,filename)])
    return filelist 
loglist=[]
for path in pathlist:
    loglist+=get_filelist(path,'.log')
print (loglist[0])
for logpath,filename,logfile in loglist[:1]:
    mol=Molnew(name=filename.strip('.log'))
    mol.Update_from_Gaulog(logpath+'/')
    mol.CalculateAtomization('HF/6-31g*')
    print ()

