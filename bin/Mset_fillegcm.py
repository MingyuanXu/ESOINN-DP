import numpy as np                     
from ESOI_HDNN_MD.Computemethod import Qmmm
from ESOI_HDNN_MD.Comparm import GPARAMS
from ESOI_HDNN_MD.Base.Info import List2str
from ESOI_HDNN_MD import UpdateGPARAMS,LoadModel
from ESOI_HDNN_MD.Train import productor,consumer
from ESOI_HDNN_MD.Train import consumer
from ESOI_HDNN_MD.Computemethod import Cal_NN_EFQ 
import os
#from TensorMol import *
from TensorMol import MSet,JOULEPERHARTREE
import argparse as arg
from multiprocessing import Queue,Process,Manager,Pool
from matplotlib import pyplot as plt 
parser=arg.ArgumentParser(description='Grep qm area from an Amber MDcrd trajory to make training dataset!')
parser.add_argument('-i','--input')

args=parser.parse_args()
jsonfile=args.input
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if __name__=="__main__":
    UpdateGPARAMS(jsonfile)
    LoadModel()
    if not os.path.exists("./results") :
        os.system("mkdir ./results")
    for i in range(len(GPARAMS.Dataset_setting.Inputdatasetlist)):
        TMPSet=MSet(GPARAMS.Dataset_setting.Inputdatasetlist[i])
        TMPSet.Load()
        for j in range(len(TMPSet.mols)):
            if j%1000==0:
                print (TMPSet.Name,j)
        #for j in range(10):
            EGCM=TMPSet.mols[j].Cal_EGCM()
        TMPSet.Save()

