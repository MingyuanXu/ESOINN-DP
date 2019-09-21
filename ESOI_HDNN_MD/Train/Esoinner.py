from __future__ import absolute_import
from __future__ import print_function
import sys
sys.path.append('../Esoinn')
import numpy as np
from parmed.amber import AmberParm
from parmed.amber import AmberMdcrd
from parmed.amber import Rst7
import pickle
import random
import argparse
from multiprocessing import Queue,Process,Manager

def ESOINNer(EsoinnQueue,if_new):
    from ..Comparm import * 
    if_continue=True
    while if_continue:
        with open(GPARAMS.Dataset_setting.ESOINNdataset,'rb') as f:
            Dataset=pickle.load(f)
        EGCM_trainingset=[]
        time=0
        while not EsoinnQueue.empty() or times<2000:
            times+=1
            q=EsoinnQueue.get()
            EGCM_trainingset+=q
            if q==None:
                if_continue=False
        EGCM_trainingset=np.array(EGCM_trainingset) 
        Dataset=np.concatenate(Dataset,EGCM_trainingset)
        GPARAMS.Esoinn_setting.Model.fit(EGCM_trainingset,iteration_steps=5000,if_reset=False)
        ESOINN_MODEL.fit(Dataset,iteration_step=50000,if_reset=False)
        GPARAMS.Esoinn_setting.Model.Save()
        pickle.save(GPARAMS.Dataset_setting.ESOINNdataset,Dataset)

