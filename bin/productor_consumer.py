import numpy as np                     
from ESOI_HDNN_MD.Computemethod import Qmmm
from ESOI_HDNN_MD.Comparm import GPARAMS
from ESOI_HDNN_MD.Base.Info import List2str
from ESOI_HDNN_MD import UpdateGPARAMS,LoadModel
from ESOI_HDNN_MD.Train import productor,consumer,esoinner,trainer,dataer
import os
#from TensorMol import *
import argparse as arg
from multiprocessing import Queue,Process,Manager,Pool
import time
#import sys
#sys.path.append("./Oldmodule")
parser=arg.ArgumentParser(description='Grep qm area from an Amber MDcrd trajory to make training dataset!')
parser.add_argument('-i','--input')

args=parser.parse_args()
jsonfile=args.input
def productors(index,QMQueue):
    print ('this is an example productor')
    return 

if __name__=="__main__":
    manager=Manager()
    QMQueue=manager.Queue()
    EsoinnQueue=manager.Queue()
    DataQueue=manager.Queue()
    GPUQueue=manager.Queue()
    if os.path.exists('./networks/lastsave'):
        os.system("rm ./networks/lastsave/* -r")
        os.system("cp *.ESOINN Sfactor.in ./networks/lastsave ")

    UpdateGPARAMS(jsonfile)
    for i in GPARAMS.Compute_setting.Gpulist:
        GPUQueue.put(i)
    for stage in range(GPARAMS.Train_setting.Trainstage,\
                       GPARAMS.Train_setting.Stagenum+GPARAMS.Train_setting.Trainstage):
        LoadModel()
        #productor(0,GPUQueue=GPUQueue)
        ProductPool=Pool(len(GPARAMS.Compute_setting.Gpulist))
        Resultlist=[]
        for i in range(len(GPARAMS.System_setting)):
            result=ProductPool.apply_async(productor,(i,QMQueue,EsoinnQueue,GPUQueue))
            Resultlist.append(result)
        ProductPool.close()
        for i in range(len(GPARAMS.System_setting)):
            tmp=Resultlist[i].get()
            print (tmp)
        Consumer_Process=Process(target=consumer,args=(QMQueue,))
        Consumer_Process.start()
        Esoinner_Process=Process(target=esoinner,args=(EsoinnQueue,))
        Esoinner_Process.start()
        ProductPool.terminate()
        ProductPool.join()
        QMQueue.put(None)
        Consumer_Process.join()
        EsoinnQueue.put(None)
        Esoinner_Process.join()

        LoadModel(ifhdnn=False)
        print ("New ESOINN model has %d clusters"%GPARAMS.Esoinn_setting.Model.class_id)
        os.system("cp *.ESOINN Sfactor.in ./networks")
        Dataer_Process=Process(target=dataer,args=(DataQueue,))
        Dataer_Process.start()
        TrainerPool=Pool(len(GPARAMS.Compute_setting.Gpulist))
        Resultlist=[]
        for i in range(GPARAMS.Esoinn_setting.Model.class_id):
            print ("Create HDNN subnet for class %d"%i)
            result=TrainerPool.apply_async(trainer,(DataQueue,GPUQueue))
            Resultlist.append(result)
        TrainerPool.close()
        for i in range(GPARAMS.Esoinn_setting.Model.class_id):
            tmp=Resultlist[i].get()
            print (tmp)
        TrainerPool.terminate()
        TrainerPool.join()
        Dataer_Process.join()
        
        if os.path.exists(GPARAMS.Compute_setting.Traininglevel):
            os.system("mkdir %s/Stage%d"%(GPARAMS.Compute_setting.Traininglevel,GPARAMS.Train_setting.Trainstage))
            os.system("mv %s/*.record Stage%d"%(GPARAMS.Compute_setting.Traininglevel,GPARAMS.Train_setting.Trainstage)) 
        for i in range(len(GPARAMS.System_setting)):
            GPARAMS.MD_setting[i].Stageindex+=1
        GPARAMS.Train_setting.Trainstage+=1
        
