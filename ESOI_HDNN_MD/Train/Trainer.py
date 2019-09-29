import numpy as np
import pickle
import random
from multiprocessing import Queue,Process,Manager
from ..Neuralnetwork import *
from ..Comparm import *
     
def NNTrainer(DataQueue,ID,ifcontinue):
    from   TensorMol import MSet,PARAMS,MolDigester
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=str(ID)
    while True:
        TMMSET,ider,maxsteps=DataQueue.get()
        if TMMSET==None:
            break
        else: 
            print('%dth GPU training Cluster %d Subnet!'%(ID,ider))
            TreatedAtoms=TMMSET.AtomTypes()
            GPARAMS.Neuralnetwork_setting.Maxsteps=maxsteps
            d=MolDigester(TreatedAtoms,name_="ANI1_Sym_Direct",OType_="EnergyAndDipole")
            tset=TData_BP_Direct_EE_WithEle(TMMSET,d,order_=1,num_indis_=1,type_="mol",WithGrad_=True,MaxNAtoms=100)
            if ifcontinue==True:
                NN_name='Cluster%d_ANI1_Sym_Direct_RawBP_EE_Charge_DipoleEncode_Update_vdw_DSF_elu_Normalize_Dropout_0'%ider
            else:
                NN_name=None
            SUBNET=BP_HDNN(tset,NN_name)
            SUBNET.train(SUBNET.max_steps,continue_training=ifcontinue)

