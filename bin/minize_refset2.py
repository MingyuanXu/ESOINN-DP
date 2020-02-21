import numpy as np                     
from ESOI_HDNN_MD.Computemethod import Qmmm
from ESOI_HDNN_MD.Comparm import GPARAMS
from ESOI_HDNN_MD.Base.Info import List2str
from ESOI_HDNN_MD import UpdateGPARAMS,LoadModel,Added_MSet
from ESOI_HDNN_MD.Train import * 
from TensorMol import MSet
import os

#from TensorMol import *
import argparse as arg
from multiprocessing import Queue,Process,Manager,Pool
import time
import random
#import sys
#sys.path.append("./Oldmodule")

parser=arg.ArgumentParser(description='Grep qm area from an Amber MDcrd trajory to make training dataset!')
parser.add_argument('-i','--input')
args=parser.parse_args()
jsonfile=args.input

if __name__=="__main__":
    manager=Manager()
    QMQueue=manager.Queue()
    DataQueue=manager.Queue()
    GPUQueue=manager.Queue()
    NetstrucQueue=manager.Queue()
    if os.path.exists('./networks/lastsave'):
        os.system("rm ./networks/lastsave/* -r")
        os.system("cp *.ESOINN Sfactor.in ./networks/lastsave ")
    UpdateGPARAMS(jsonfile)
    for i in GPARAMS.Compute_setting.Gpulist:
        GPUQueue.put(i)

    bigset=MSet('Bigset')
#    GPARAMS.Dataset_setting.Inputdatasetlist=random.sample(GPARAMS.Dataset_setting.Inputdataset
#    for name in GPARAMS.Dataset_setting.Inputdatasetlist:
#        tmpset=MSet(name)
#        tmpset.Load()
#        bigset.mols+=tmpset.mols
#    for i in range(GPARAMS.Compute_setting.Checkernum):
#        checker_set=MSet('Bigset_%d'%i)
#        checker_set.mols=[bigset.mols[j] for j in range(len(bigset.mols)) if j%(i+1)==0]
#        checker_set.mols=[bigset.mols[0]]+random.sample(checker_set.mols,min(GPARAMS.Compute_setting.Checkerstep,len(checker_set.mols)))
#        checker_set.Save()
#    bigset=None 
    for stage in range(GPARAMS.Train_setting.Trainstage,\
                       GPARAMS.Train_setting.Stagenum+GPARAMS.Train_setting.Trainstage):
        LoadModel()
        #==Main MD process with productor and Consumer model==
        ProductPool=Pool(len(GPARAMS.Compute_setting.Gpulist))
        Resultlist=[]
        for i in range(len(GPARAMS.System_setting)):
            result=ProductPool.apply_async(productor,(i,QMQueue,GPUQueue))
            Resultlist.append(result)
        for i in range(GPARAMS.Compute_setting.Checkernum):
            result=ProductPool.apply_async(remover,(i,QMQueue,GPUQueue))
            Resultlist.append(result)
        ProductPool.close()
        for i in range(len(GPARAMS.System_setting)):
            tmp=Resultlist[i].get()
            print (tmp)
        for i in range(GPARAMS.Compute_setting.Checkernum):
            tmp=Resultlist[i].get()
            print (tmp)
        Consumer_Process=Process(target=consumer,args=(QMQueue,))
        Consumer_Process.start()
        ProductPool.terminate()
        ProductPool.join()
        QMQueue.put(None)
        Consumer_Process.join()
        #==parallel Mol caclulator==
        parallel_caljob("Stage_%d_Newadded"%GPARAMS.Train_setting.Trainstage,manager,ctrlfile=jsonfile)
        #==Esoi-layer Training process==
        Added_MSet("Stage_%d_Newadded"%GPARAMS.Train_setting.Trainstage)
        esoinner()         
        LoadModel(ifhdnn=False)
        print ("New ESOINN model has %d clusters"%GPARAMS.Esoinn_setting.Model.class_id)
        os.system("cp *.ESOINN Sfactor.in ./networks")
        if os.path.exists(GPARAMS.Compute_setting.Traininglevel):
            os.system("mkdir %s/Stage%d"%(GPARAMS.Compute_setting.Traininglevel,GPARAMS.Train_setting.Trainstage))
        Dataer_Process=Process(target=dataer,args=(DataQueue,))
        Dataer_Process.start()
        if GPARAMS.Train_setting.Ifgpuwithhelp==True:
            TrainerPool=Pool(max(GPARAMS.Esoinn_setting.Model.class_id+1,GPARAMS.Train_setting.Modelnumperpoint+1))
        else:
            TrainerPool=Pool(len(GPARAMS.Compute_setting.Gpulist))
        Resultlist=[]
        for i in range(max(GPARAMS.Esoinn_setting.Model.class_id,GPARAMS.Train_setting.Modelnumperpoint)):
            print ("Create HDNN subnet for class %d"%i)
            result=TrainerPool.apply_async(trainer,(DataQueue,GPUQueue,jsonfile))
            Resultlist.append(result)
        if GPARAMS.Esoinn_setting.Loadrespnet==True and GPARAMS.Esoinn_setting.Ifresp==True:
            result=TrainerPool.apply_async(respnet_train,("HF_resp",GPUQueue,jsonfile))
            Resultlist.append(result)
        TrainerPool.close()
        for i in range(max(GPARAMS.Esoinn_setting.Model.class_id,GPARAMS.Train_setting.Modelnumperpoint+1)):
            tmp=Resultlist[i].get()
            print (tmp)
        if GPARAMS.Esoinn_setting.Loadrespnet==True and GPARAMS.Esoinn_setting.Ifresp==True:
            tmp=Resultlist[i].get()
            print (tmp)
        TrainerPool.terminate()
        TrainerPool.join()
        Dataer_Process.join()
        if os.path.exists(GPARAMS.Compute_setting.Traininglevel):
            os.system("mv %s/*.record %s/Stage%d"%(GPARAMS.Compute_setting.Traininglevel,\
                                                   GPARAMS.Compute_setting.Traininglevel,\
                                                   GPARAMS.Train_setting.Trainstage)) 
        LoadModel()
        RMSE=evaler()
        GPARAMS.Train_setting.rmse=RMSE[0]
        print ("NN result: ",RMSE)
        for i in range(len(GPARAMS.System_setting)):
            GPARAMS.MD_setting[i].Stageindex+=1
        GPARAMS.Train_setting.Trainstage+=1
        
