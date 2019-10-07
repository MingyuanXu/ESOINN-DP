import numpy as np                     
from ESOI_HDNN_MD.Computemethod import Qmmm
from ESOI_HDNN_MD.Comparm import GPARAMS
from ESOI_HDNN_MD.Base.Info import List2str
from ESOI_HDNN_MD import UpdateGPARAMS
from ESOI_HDNN_MD.Train import productor,consumer
from ESOI_HDNN_MD.Train import consumer
import os
#from TensorMol import *
import argparse as arg
from multiprocessing import Queue,Process,Manager,Pool

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
    UpdateGPARAMS(jsonfile)
#    """
    productor(0,QMQueue)
    QMQueue.put(None)
    consumer(QMQueue)
    """
    for stage in range(GPARAMS.Train_setting.Trainstage,\
                       GPARAMS.Train_setting.Stagenum+GPARAMS.Train_setting.Trainstage):


        ProductPool=Pool(len(GPARAMS.Compute_setting.Gpulist))
        for i in range(len(GPARAMS.System_setting)):
            ProductPool.apply_async(productor,(i,QMQueue))
        ProductPool.close()
        Consumer_Process=Process(target=consumer,args=(QMQueue,))
        Consumer_Process.start()
        ProductPool.join()
        QMQueue.put(None)
        Consumer_Process.join()
        for i in range(len(GPARAMS.System_setting)):
            GPARAMS.MD_setting[i].Stageindex+=1
        GPARAMS.Train_setting.Trainstage+=1


    """
