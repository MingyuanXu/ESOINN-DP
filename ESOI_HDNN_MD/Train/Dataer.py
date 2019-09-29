import numpy as np
import pickle
import random
import argparse
from multiprocessing import Queue,Process,Manager
import os 
from ..Comparm import *
def Dataer(Dataqueue,GPUNum):
    from TensorMol import MSet
    Trainingset=MSet(GPARAMS.Compute_setting.Traininglevel)
    Trainingset.Load()
    Newadded_Set=MSet('Newadded')
    if os.path.exists('./datasets/Newadded.pdb'):
        Newadded_Set.Load()
    ClusNum=GPARAMS.Compute_setting.zeros.class_id
    CluNmols_before=np.zeros(ClusNum)
    SubTrainList=[]
    for i in range(ClusNum):
        SubTrainSet=MSet(GPARAMS.Esoinn_setting.Traininglevel+'_Cluster%d'%i)
        SubTrainList.append(SubTrainSet)
    print('start make cluster for training set')
    for i in range(len(Trainingset.mols)):
        if i%1000==0:
            print (i)
        EGCM=(Trainingset.mols[i].EGCM-GPARAMS.Esoinn_setting.Scalemin)/\
                (GPARAMS.Esoinn_setting.Scalemax-GPARAMS.Esoinn_setting.Scalemin)
        EGCM[ ~ np.isfinite( EGCM )] = 0
        list=GPARAMS.Esoinn_setting.Model.find_closest_cluster(3,EGCM)
        for j in list:
            SubTrainList[j].mols.append(Trainingset.mols[i])
    for i in range(ClusNum):
        othermollist=[]
        for j in range(ClusNum):
            if j!=i:
                othermollist+=SubTrainList[j].mols 
        SubTrainList[i].mols+=random.sample(othermollist,\
                                   (len(Trainingset.mols)-len(SubTrainList[i].mols))*GPARAMS.Esoinn_setting.Mixrate)
        SubTrainList.Save()
    NewTrainList=[]
    for i in range(ClusNum):
        NewSet=MSet(GPARAMS.Esoinn_setting.Traininglevel+'_New%d'%i)
        NewTrainList.append(NewSet)
    for i in range(len(Newadded_Set.mols)):
        if i%1000==0:
            print (i)
        EGCM=(Newadded_Set.mols[i].EGCM-GPARAMS.Esoinn_setting.Scalemin)/\
                (GPARAMS.Esoinn_setting.Scalemax-GPARAMS.Esoinn_setting.Scalemin)
        EGCM[ ~ np.isfinite( EGCM )] = 0
        list=GPARAMS.Esoinn_setting.find_closest_cluster(3,EGCM)
        for j in list:
            NewTrainList[j].mols.append(Newadded_Set.mols[i]) 
    Newadded_Num=[len(m.mols) for m in NewTrainList ]

    for i in range(ClusNum):
        tmp=MSet('tmpset') 
        Num=min(8000,max(Newadded_Num[i],2000))
        print (i,Num,len(SubTrainList[i].mols))
        flag=False
        while not flag:
            tmp.mols=random.sample(SubTrainList[i].mols,Num)
            flag=(np.array(tmp.AtomTypes())==np.array(SubTrainList[i].AtomTypes())).all()
        NewTrainList[i].mols+=tmp.mols    
        if GPARAMS.Train_setting.Ifwithhelp==True:
            NewTrainList[i].Save()
            
    for i in range(ClusNum):
        if Newadded_Num[i]>0:
            if Newadded_Num[i]>20000:
                maxsteps=Newadded_Num[i]/40*10
            elif Newadded_Num[i]<=20000 and Newadded_Num[i]>=8000:
                maxsteps=10000
            else:
                maxsteps=6000
            Dataqueue.put((NewTrainList[i],i,maxsteps))
            print ('%dth cluster is put in queue, mol num: %d!'%(i,len(NewTrainList[i].mols)))
    for j in range(GPUNum):
        Dataqueue.put((None,j,0))
     
