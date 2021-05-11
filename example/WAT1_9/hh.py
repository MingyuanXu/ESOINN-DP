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
    TMPSet=MSet("PM6_Cluster0")
    TMPSet.Load()
        
            
    print (TMPSet.mols[0],TMPSet.mols[0].properties)
