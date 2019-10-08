import numpy as np
import pickle
import random
import argparse
from multiprocessing import Queue,Process,Manager
import os 
from ..Comparm import *
def Check_MSet(mollist):
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
        force=i.properties['force']
        try:
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
                print (i.name,i,mindis,maxdis,maxforce,len(force),len(i.properties['dipole']))
            except:
                print (i.name,i,mindis,maxdis)
    return mols 
        
def dataer(Dataqueue):
    from TensorMol import MSet
    Trainingset=MSet(GPARAMS.Compute_setting.Traininglevel)
    Trainingset.Load()
    Trainingset.mols=Check_MSet(Trainingset.mols)
    print ("Trainingset.mols :",len(Trainingset.mols))
    Newadded_Set=MSet('Newadded')
    if os.path.exists('./datasets/Newadded.pdb'):
        Newadded_Set.Load()
        Newadded_Set.mols=Check_MSet(Newadded_Set.mols)
        print ("Newadded_Set.mols :",len(Newadded_Set.mols))
    ClusNum=GPARAMS.Esoinn_setting.Model.class_id
    CluNmols_before=np.zeros(ClusNum)
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
                    
        list=GPARAMS.Esoinn_setting.Model.find_closest_cluster(min(GPARAMS.Train_setting.Modelnumperpoint,GPARAMS.Esoinn_setting.Model.class_id),EGCM)
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

    NewTrainList=[]
    for i in range(ClusNum):
        NewSet=MSet(GPARAMS.Compute_setting.Traininglevel+'_New%d'%i)
        NewTrainList.append(NewSet)
    for i in range(len(Newadded_Set.mols)):
        try:
            EGCM=(Newadded_Set.mols[i].EGCM-GPARAMS.Esoinn_setting.scalemin)/\
                (GPARAMS.Esoinn_setting.scalemax-GPARAMS.Esoinn_setting.scalemin)
        except:
            EGCM=(Newadded_Set.mols[i].Cal_EGCM()-GPARAMS.Esoinn_setting.scalemin)/\
                (GPARAMS.Esoinn_setting.scalemax-GPARAMS.Esoinn_setting.scalemin)
                
        EGCM[ ~ np.isfinite( EGCM )] = 0
        list=GPARAMS.Esoinn_setting.Model.find_closest_cluster(min(GPARAMS.Train_setting.Modelnumperpoint,GPARAMS.Esoinn_setting.Model.class_id),EGCM)
        for j in list:
            NewTrainList[j].mols.append(Newadded_Set.mols[i]) 
    Newadded_Num=[len(m.mols) for m in NewTrainList ]

    for i in range(ClusNum):
        tmp=MSet('tmpset') 
        if len(SubTrainList[i].mols)<=GPARAMS.Train_setting.Samplecontrol[0]:
            Num=len(SubTrainList[i].mols)
        else:
            Num=min(GPARAMS.Train_setting.Samplecontrol[1],\
                    max(Newadded_Num[i],GPARAMS.Train_setting.Samplecontrol[0]))
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
            if Newadded_Num[i]>GPARAMS.Train_setting.Maxbatchnumpertrain[-1]:
                maxsteps=Newadded_Num[i]/GPARAMS.Neuralnetwork_setting.Batchsize*GPARAMS.Train_setting.Maxepochpertrain 
            elif Newadded_Num[i]<=GPARAMS.Train_setting.Batchnumcontrol[-1] and\
                    Newadded_Num[i]>=GPARAMS.Train_setting.Batchnumcontrol[0]:
                maxsteps=GPARAMS.Train_setting.Maxbatchnumpertrain[-1]
            else:
                maxsteps=GPARAMS.Train_setting.Maxbatchnumpertrain[0]
            Dataqueue.put((NewTrainList[i],i,maxsteps))
            print ('%dth cluster is put in queue, mol num: %d!'%(i,len(NewTrainList[i].mols)))

    for j in range(len(GPARAMS.Compute_setting.Gpulist)):
        Dataqueue.put((None,j,0))
     
