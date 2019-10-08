import numpy as np
import pickle
import random
from multiprocessing import Queue,Process,Manager
from ..Neuralnetwork import *
from ..Comparm import *
     
def trainer(DataQueue,GPUQueue=None):
#def trainer(TMMSET,ider,maxsteps,GPUQueue):
    from   TensorMol import MSet,PARAMS,MolDigester
    import os
    from ..Base import Find_useable_gpu
    GPUid=GPUQueue.get()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(GPUid )
#    os.environ["CUDA_VISIBLE_DEVICES"]=Find_useable_gpu(GPARAMS.Compute_setting.Gpulist)
    print ("Visible CPU ID: %s training Cluster"\
                   %(os.environ["CUDA_VISIBLE_DEVICES"]))
    TMMSET,ider,maxsteps=DataQueue.get()
    print ("Visible CPU ID: %s training Cluster %d subnet"\
           %(os.environ["CUDA_VISIBLE_DEVICES"],ider))
    try:
        TreatedAtoms=TMMSET.AtomTypes()
        GPARAMS.Neuralnetwork_setting.Maxsteps=maxsteps
        d=MolDigester(TreatedAtoms,name_="ANI1_Sym_Direct",OType_="EnergyAndDipole")
        tset=TData_BP_Direct_EE_WithEle(TMMSET,d,order_=1,num_indis_=1,type_="mol",WithGrad_=True,MaxNAtoms=100)
        if GPARAMS.Train_setting.Trainstage==0:
            ifcontinue=False 
        else:
            ifcontinue=True
        if ifcontinue==True:
            NN_name=GPARAMS.Esoinn_setting.efdnetname+'%d_ANI1_Sym_Direct_RawBP_EE_Charge_DipoleEncode_Update_vdw_DSF_elu_Normalize_Dropout_0'%ider
        else:
            NN_name=None
        GPARAMS.Neuralnetwork_setting.Maxsteps=maxsteps 
        SUBNET=BP_HDNN(tset,NN_name)
        SUBNET.train(SUBNET.max_steps,continue_training=ifcontinue)
    except:
        Print("Trainer Process %d GPUID %d is wrong!"%(ider,GPUid))
    GPUQueue.put(GPUid)

