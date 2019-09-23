import numpy as np
from parmed.amber import AmberParm
from parmed.amber import AmberMdcrd
from parmed.amber import Rst7
from ESOI_HDNN_MD.Computemethod import Qmmm
from ESOI_HDNN_MD.Comparm import GPARAMS
from ESOI_HDNN_MD.Base.Info import List2str
from ESOI_HDNN_MD import UpdateGPARAMS
from ESOI_HDNN_MD.Train import productor
import os
from TensorMol import *
import argparse as arg
from multiprocessing import Queue,Process,Manager,Pool

#import sys
#sys.path.append("./Oldmodule")

parser=arg.ArgumentParser(description='Grep qm area from an Amber MDcrd trajory to make training dataset!')
parser.add_argument('-i','--input')
args=parser.parse_args()
jsonfile=args.input
def ProducPool(Queue,processnum):
    pool=Pool(processnum)
    for i in range(len(GPARAMS.System_setting)):
        if len(GPARAMS.Compute_setting.Gpulist)==0:
            workid=""
        else:
            workid=str(math.ceil)
        pool.apply_async(productor,(workid,Queue,))
    
if __name__=="__main__":
    manager=Manager()
    QMQueue=Queue()
    EsoinnQueue=Queue()
    DataQueue=Queue()
     
    UpdateGPARAMS(jsonfile)
    for stage in range(GPARAMS.Train_setting.Trainstage,\
                        GPARAMS.Train_setting.Stagenum+GPARAMS.Train_setting.Trainstage)
        for i in range(len(GPARAMS.System_setting)):
            Produc_Process=Process(targer=productor,agrs=(i,))
            productor(ID="0",GPARAMS_index=0,Queue=None)   
     
