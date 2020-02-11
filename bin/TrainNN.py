#! /usr/bin/env python3
from multiprocessing import Queue,Process,Manager
from TensorMol import  *
from ESOI_HDNN_MD import *
from ESOI_HDNN_MD import UpdateGPARAMS,GPARAMS 
from ESOI_HDNN_MD.Train import calculator 
from ESOI_HDNN_MD.Base import Find_useable_gpu 
from ESOI_HDNN_MD.Neuralnetwork import *
import os
import math
import argparse as arg
import time
import random

parser=arg.ArgumentParser(description="Training Neural Network Potentials for MSet")
parser.add_argument('-i',"--ctrlfile")
parser.add_argument('-d',"--dataset")
parser.add_argument('-s',"--struc")
parser.add_argument('-t',"--type")
args=parser.parse_args()
time.sleep(random.randint(1,10)*20)
UpdateGPARAMS(args.ctrlfile)
evostruc=[int(i) for i in args.struc.split('_')]
print (evostruc)
TMPset=MSet(args.dataset)
TMPset.Load()
if len(TMPset.mols)< GPARAMS.Neuralnetwork_setting.Batchsize*20 :
    num=math.ceil(GPARAMS.Neuralnetwork_setting.Batchsize*20/len(TMPset.mols))
    TMPset.mols=TMPset.mols*num
GPUID=Find_useable_gpu([0,1,2,3,4,5,6,7,8])
os.environ["CUDA_VISIBLE_DEVICES"]=str(GPUID)
TreatedAtoms=TMPset.AtomTypes()
d=MolDigester(TreatedAtoms,name_="ANI1_Sym_Direct",OType_="EnergyAndDipole")
if args.type=="bpresp":
    GPARAMS.Neuralnetwork_setting.Switchrate=0.9
    tset=TData_BP_Direct_EE_WithCharge(TMPset,d,order_=1,num_indis_=1,type_="mol",WithGrad_=True,MaxNAtoms=100)
else:
    tset=TData_BP_Direct_EE_WithEle(TMPset,d,order_=1,num_indis_=1,type_="mol",WithGrad_=True,MaxNAtoms=100)
NN_name=None 
ifcontinue=False

if args.type=="bpresp":
    SUBNET=BP_HDNN_charge(tset,NN_name,Structure=evostruc)
    Ncase,batchnumf,Lossf,Losse,batchnumd,Lossd,structure=SUBNET.train(SUBNET.max_steps,continue_training=ifcontinue)
else:
    SUBNET=BP_HDNN(tset,NN_name,Structure=evostruc)
    Ncase,batchnumf,Lossf,Losse,batchnumd,Lossd,structure=SUBNET.train(SUBNET.max_steps,continue_training=ifcontinue,AimF=GPARAMS.Neuralnetwork_setting.Aimf)

if args.type!="bpresp":
    strucstr=" ".join([str(i) for i in structure])
    NNstrucfile=open(GPARAMS.Neuralnetwork_setting.NNstrucrecord,'w')
    NNstrucfile.write("%d, %d, %f, %f, %d, %f, %s,\n"\
            %(Ncase,batchnumf,Lossf,Losse,batchnumd,Lossd,strucstr))
    """

if args.type=="bp":
    SUBNET=BP_HDNN(None,)
    Etotal,Ebp,Ebp_atom,Ecc,Evdw,mol_dipole,atom_charge,gradient=EvalSet(TMMSET)
    
    """
