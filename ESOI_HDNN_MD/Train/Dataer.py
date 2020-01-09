import numpy as np
import pickle
import random
import argparse
from multiprocessing import Queue,Process,Manager
import os 
from ..Comparm import *
def Check_MSet(mollist,level=0):
    mols=[]
    for i in mollist:
        flag=True
        natom=len(i.atoms)
        crd=i.coords
        maxdis=0;mindis=10
        for j in range(0,natom-1):
            for k in range(j+1,natom):
                dis=np.sqrt(np.sum((crd[j]-crd[k])**2))
                if dis>maxdis:
                    maxdis=dis
                if dis<mindis:
                    mindis=dis
        if mindis<0.75 or maxdis>20:
            flag=False
        if level>0:
            try:
                force=i.properties['force']
                length=len(force)
                maxforce=np.max(np.abs(force))*627.51
                if length!=natom or maxforce>400:
                    flag=False
                if len(i.properties["dipole"] )!=3:
                    flag=False
            except:
                flag=False
        if flag==True:
            mols.append(i)
        else:
            try:
                print (i.name,mindis,maxdis,maxforce,len(force),len(i.properties['dipole']))
            except:
                print (i.name,mindis,maxdis)
    return mols 
        
def dataer(Dataqueue):
    from TensorMol import MSet
    Trainingset=MSet(GPARAMS.Compute_setting.Traininglevel)
    Trainingset.Load()
    Trainingset.mols=Check_MSet(Trainingset.mols,level=1)
    Trainingset.Save()
    respset=MSet('HF_resp')
    respset.Load()
    respset.mols=Check_MSet(respset.mols,level=1)
    respset.Save()
    print ("Trainingset.mols :",len(Trainingset.mols))
    ClusNum=max(GPARAMS.Esoinn_setting.Model.class_id,GPARAMS.Train_setting.Modelnumperpoint)
    print ("++++++++++++++++++Dataer++++++++++++++++++++++")
    print ("ClusNum:",ClusNum)
    SubTrainList=[]
    for i in range(ClusNum):
        SubTrainSet=MSet(GPARAMS.Compute_setting.Traininglevel+'_Cluster%d'%i)
        SubTrainList.append(SubTrainSet)
    print('start make cluster for training set')
    for i in range(len(Trainingset.mols)):
        try:
            EGCM=(Trainingset.mols[i].EGCM-GPARAMS.Esoinn_setting.scalemin)/\
                    (GPARAMS.Esoinn_setting.scalemax-GPARAMS.Esoinn_setting.scalemin)
        except:
            EGCM=(Trainingset.mols[i].Cal_EGCM()-GPARAMS.Esoinn_setting.scalemin)/\
                    (GPARAMS.Esoinn_setting.scalemax-GPARAMS.Esoinn_setting.scalemin)
        EGCM[ ~ np.isfinite( EGCM )] = 0
        if GPARAMS.Esoinn_setting.Model.class_id >=GPARAMS.Train_setting.Modelnumperpoint:            
            list=GPARAMS.Esoinn_setting.Model.find_closest_cluster(min(GPARAMS.Train_setting.Modelnumperpoint,GPARAMS.Esoinn_setting.Model.class_id),EGCM)
        else:
            list=[i for i in range(GPARAMS.Train_setting.Modelnumperpoint)]
        for j in list:
            SubTrainList[j].mols.append(Trainingset.mols[i])

    for i in range(ClusNum):
        print ("Cluster %d has %d mols"%(i,len(SubTrainList[i].mols)))

    for i in range(ClusNum):
        othermollist=[]
        for j in range(ClusNum):
            if j!=i and len(SubTrainList[j].mols)>2:
                othermollist+=SubTrainList[j].mols 
        print ("Other mol list for Cluster %d"%i,len(othermollist))
        if len(othermollist) >0:
            samplenum=min(\
                          math.ceil((len(Trainingset.mols)-len(SubTrainList[i].mols))*GPARAMS.Esoinn_setting.Mixrate),\
                          len(othermollist)\
                         )
            print (len(othermollist),samplenum)
            SubTrainList[i].mols+=random.sample(othermollist,samplenum)
        SubTrainList[i].Save()
    
    for i in range(ClusNum):
        Dataqueue.put((SubTrainList[i],i,GPARAMS.Train_setting.Maxsteps))
        print ('%dth cluster is put in queue, mol num: %d!'%(i,len(SubTrainList[i].mols)))
#    for j in range(len(GPARAMS.Compute_setting.Gpulist)):
#        Dataqueue.put((None,j,0))
     
