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
from ..Comparm import GPARAMS 

def ESOINNer(EsoinnQueue):
    if_continue=True
    cluster_center_before=GPARAMS.Esoinn_setting.Model.cal_cluster_center()
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
        Noiseset,_=GPARAMS.Esoinn_setting.Model.predict(Dataset)
        GPARAMS.Esoinn_setting.Model.fit(Noiseset,iteration_steps=50000,if_reset=False)
        GPARAMS.Esoinn_setting.Model.Save()
        pickle.save(GPARAMS.Dataset_setting.ESOINNdataset,Dataset)
    cluster_center_after=GPARAMS.Esoinn_setting.Model.cal_cluster_center()
    updaterule=np.zeros(GPARAMS.Esoinn_setting.Modelfile.E)
    for i in range(len(cluster_center_after)):
        vec1=cluster_center_after[i]
        dis=np.sum(np.array(cluster_center_before)-np.array([vec1]*len(cluster_center_before))**2,1) 
        index=np.argmin(dis)[0]
        print (i,vec1)

