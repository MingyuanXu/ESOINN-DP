import numpy as np                     
from ESOI_HDNN_MD.Computemethod import Qmmm
from ESOI_HDNN_MD.Comparm import GPARAMS
from ESOI_HDNN_MD.Base.Info import List2str
from ESOI_HDNN_MD import UpdateGPARAMS,LoadModel
from ESOI_HDNN_MD.Train import productor,consumer,esoinner,trainer,dataer,esoinn_train 
import os
from TensorMol import *
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
    DataQueue=manager.Queue()
    GPUQueue=manager.Queue()

    UpdateGPARAMS(jsonfile)
    for i in GPARAMS.Compute_setting.Gpulist:
        GPUQueue.put(i)
    for stage in range(GPARAMS.Train_setting.Trainstage,\
                       GPARAMS.Train_setting.Stagenum+GPARAMS.Train_setting.Trainstage):

        LoadModel()
        esoinn_train()
        LoadModel(ifhdnn=False)
        print ("New ESOINN model has %d clusters"%GPARAMS.Esoinn_setting.Model.class_id)
        Dataer_Process=Process(target=dataer,args=(DataQueue,))
        Dataer_Process.start()
        TrainerPool=Pool(len(GPARAMS.Compute_setting.Gpulist))
        for i in range(GPARAMS.Esoinn_setting.Model.class_id):
            print ("Create HDNN subnet for class %d"%i)
            TrainerPool.apply_async(trainer,(DataQueue,GPUQueue))
        TrainerPool.close()
        TrainerPool.join()
        Dataer_Process.join()
        """
        TMMSet=MSet('PM6_New0')
        TMMSet.Load()
        ider=0
        maxsteps=2000
        trainer(TMMSet,ider,maxsteps,GPUQueue)
        """
        for i in range(len(GPARAMS.System_setting)):
            GPARAMS.MD_setting[i].Stageindex+=1
        GPARAMS.Train_setting.Trainstage+=1
        
