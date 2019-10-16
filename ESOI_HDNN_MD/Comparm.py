#!/usr/bin/env python
# coding=utf-8

import json
import numpy as np
from itertools import product
import os 
import pickle
from TensorMol import * # it will be remove soon

Element_Table={30:'Zn',6:'C',8:'O',1:'H',7:'N',15:'P',16:'S',12:'Mg',20:'Ca',80:'Hg',29:'Cu'}
Table_Element={'ZN':30,'Zn':30,'C':6,'O':8,'H':1,'N':7,'P':15,'S':16,'Mg':12,'Ca':20,'Hg':80,'Cu':29}
Lettertable={1:'A',2:'B',3:'C',4:'D',5:'E',6:'F',7:'G',8:'H',9:'I',10:'J',11:'K'}

class Esoinn_setting:
    def __init__(self):
        self.Atype=[]
        self.Amax=[]
        self.Modelfile=''
        self.Scalefactorfile=''
        self.Target="Train" #'Train','Predict'
        self.Maxsteps=0
        self.Traininterval=500
        self.Clusterinterval=50000
        self.Loadefdnet=False 
        self.Loadrespnet=False  
        self.efdnetname=""
        self.respnetname=""
        self.NNdict={} 
        self.scalemax=None
        self.scalemin=None
        self.Mixrate=0.1
        self.Model=None 
        return
    def Update(self):
        self.Maxnum=np.sum(self.Amax)
        self.Apt=np.zeros((len(self.Atype),2),dtype=int)
        for i in range(len(self.Atype)):
            if i>=1:
                self.Apt[i][0]=self.Apt[i-1][1]+1
            self.Apt[i][1]=self.Apt[i][0]+self.Amax[i]-1
        self.Eindex=np.zeros(self.Maxnum,dtype=int)
        for i,j in product(range(self.Maxnum),range(len(self.Atype))):
            if i >=self.Apt[j][0] and i <=self.Apt[j][1]:
                self.Eindex[i]=self.Atype[j]
        return
    
class Compute_setting:
    def __init__(self):
        self.Gpulist=[]
        self.Theroylevel=''    #Semiqm,DFT,MM,NN,REACXFF
        self.Computelevel=[""] #Full,QM/MM,Fragscheme
        self.Traininglevel=""
        self.Ompthreads=28
        self.Qmradius=2.8
        self.Ncoresperthreads=1
        self.Gaussiankeywords=""
        self.Atomizationlevel=""
        self.Consumerprocessnum=1
        return
    def Update(self):
        if not os.path.exists(self.Traininglevel):
            os.system("mkdir "+self.Traininglevel)
        return

class Software_setting:
    def __init__(self):
        self.Name="Gaussian"
        self.G16path=''
        self.Dftbpath=''
        self.Lammpspath=''
        self.Dftbparapath=''
        with os.popen('which g16','r') as f:
            for eachline in f:
                if 'g16' in eachline:
                    self.G16path=eachline.strip()
                    print (self.G16path)
        with os.popen('which dftb+','r') as f:
            for eachline in f:
                if 'dftb+' in eachline:
                    self.Dftbpath=eachline.strip()
        with os.popen('which lmp','r')  as f:
            for eachline in f:
                if 'lmp' in eachline:
                    self.Lammpspath=eachline.strip()
        return 

class System_setting:
    def __init__(self):
        self.Forcefield='Amber'
        self.Systemparm=""
        self.Strucdict={"NCHAIN":0,"CHNPTS":0,"CHNPTE":0,"CENTER":0}
        self.Traj=""
        self.Initstruc=""
        self.resptrace=[]
        self.reportcharge=[]
        return 

class MD_setting:
    def __init__(self):
        self.Mdformat='Amber'
        self.Mdout='MD.out'
        self.Temp=300
        self.Thermostat='Anderson'
        self.Center=0
        self.Mddt=1
        self.Mdv0='Thermostat'
        self.Mdmaxsteps=10
        self.Mdrestart=False
        self.Mdstage=0
        self.Nprint=10
        self.Capradius=0
        self.Capf=50
        self.Icap=False
        self.Ibox=False 
        self.Boxradius=0
        self.Stageindex=0 
        self.Mode="Train"
        self.MDmethod="Normal MD"
        self.Name="MD"
        self.Box=np.zeros(3)
        return
    def Update():
        if not os.path.exists(self.Name):
            os.system('mkdir '+self.Name)
        return

class Dataset_setting:
    def __init__(self):
        self.Outputdataset="Output.pdb"
        self.Inputdatasetlist=[]
        self.ESOINNdataset="list.EGCM"
        return 

class Train_setting:
    def __init__(self):
        self.Ifwithhelp=False
        self.Trainstage=0
        self.Stagenum=1
        #self.Maxepochpertrain=10
        #self.Maxbatchnumpertrain=[6000,10000]
        #self.Batchnumcontrol=[8000,20000]
        self.Modelnumperpoint=3
        #self.Samplecontrol=[2000,8000]
        self.Esoistep=50000
        self.Maxsteps=10000
        return 

class Neuralnetwork_setting:
    def __init__(self):
        self.Momentum=0.95;                     self.Batchsize=40
        self.Testfreq=1;                        self.tfprec="tf.float64"
        self.Scalar={"E":1,"F":0.05,"D":1}
        self.Neuraltype="sigmoid_with_param";    self.Sigmoidalpha=100.0
        self.Initstruc=[160,80,40]
        self.EEcutoff=15.0
        self.EEcutoffon=0
        self.Eluwidth=4.6
        self.EEcutoffoff=15.0
        self.DSFAlpha=0.18
        self.AddEcc=True
        self.Keepprob=[1.0,1.0,1.0,0.7]
        self.Learningrate=[0.001,0.0001,0.00001]
        self.Learningrateboundary=[0.2,0.5]
        self.Switchrate=0.5
        self.Networkprefix="./networks/"
        self.Maxcheckpoints=1
        self.Innormroutine=None
        self.Outnormroutine=None
        self.Monitorset=None
        self.AN1_r_Rc=4.6
        self.AN1_a_Rc=3.1
        self.AN1_eta=4.0
        self.AN1_zeta=8.0
        self.AN1_num_r_Rs=32
        self.AN1_num_a_Rs=8
        self.AN1_num_a_As=8
        self.AN1_r_Rs= np.array([ self.AN1_r_Rc*i/self.AN1_num_r_Rs for i in range (0, self.AN1_num_r_Rs) ])
        self.AN1_a_Rs= np.array([ self.AN1_a_Rc*i/self.AN1_num_a_Rs for i in range (0, self.AN1_num_a_Rs) ])
        self.AN1_a_As=np.array([ 2.0*Pi*i/self.AN1_num_a_As for i in range (0, self.AN1_num_a_As) ]) 
        self.Profiling=False
        self.Classify=False
        self.Maxtimeperelement=36000
        self.Maxmemperelement=16000
        self.Chopto=False
        self.Testratio=0.1
        self.Randomizedata=True
        self.Maxerr=25 #kcal/mol/angstrom
        self.Miderr=9
        self.Midrate=0.3
        self.NNstrucrecord=""
        self.NNstrucselect=[]
        self.Aime=0.002
        self.Aimf=0.005
        self.Aimd=0.05
        return
    def Update(self):
        if not os.path.exists(self.Networkprefix):
            os.system('mkdir '+self.Networkprefix)
        #this function will be remove soon!
        PARAMS["momentum"]=self.Momentum    ;   PARAMS["tf_prec"]=self.tfprec
        PARAMS["batch_size"]=self.Batchsize ;   PARAMS["test_freq"]=self.Testfreq
        PARAMS["EnergyScalar"]=self.Scalar["E"];PARAMS["GradScalar"]=self.Scalar["F"]
        PARAMS["DipoleScalar"]=self.Scalar["D"];PARAMS["NeuronType"]=self.Neuraltype
        PARAMS["HiddenLayers"]=self.Initstruc;  PARAMS["EECutoff"]=self.EEcutoff
        PARAMS["EECutoffOn"]=self.EEcutoffon;   PARAMS["EECutoffOff"]=self.EEcutoffoff
        PARAMS["DSFAlpha"]=self.DSFAlpha;       PARAMS["AddEcc"]=self.AddEcc
        PARAMS["KeepProb"]=self.Keepprob;       
        PARAMS["networks_directory"]=self.Networkprefix; PARAMS["max_checkpoints"] =self.Maxcheckpoints
        PARAMS["InNormRoutine"]=self.Innormroutine;PARAMS["OutNormRoutine"]=self.Outnormroutine
        PARAMS["MonitorSet"]=self.Monitorset;   PARAMS["AN1_r_Rc"]=self.AN1_r_Rc
        PARAMS["AN1_a_Rc"]=self.AN1_a_Rc;       PARAMS["AN1_eta"]=self.AN1_eta
        PARAMS["AN1_zeta"]=self.AN1_zeta;       PARAMS["AN1_num_r_Rs"]=32
        PARAMS["AN1_num_a_Rs"]=self.AN1_num_a_Rs; PARAMS["AN1_num_a_As"]=8
        PARAMS["Profiling"]=self.Profiling;     PARAMS["Classify"]=self.Classify
        PARAMS["MxTimePerElement"]=self.Maxtimeperelement;
        PARAMS["MxMemPerElement"]=self.Maxmemperelement;
        PARAMS["ChopTo"]=self.Chopto;           PARAMS["TestRatio"]=self.Testratio
        PARAMS["RandomizeData"]=self.Randomizedata
        if self.NNstrucrecord!="" and os.path.exists(self.NNstrucrecord):
            file=open(self.NNstrucrecord,'r')
            for eachline in file:
                var=eachline.strip().split()
                if len(var)!=0:
                    print (var)
                    ncase=int(var[0])
                    batchnum=[int(var[1]),int(var[4])]
                    loss=[float(var[2]),float(var[3]),float(var[5])]
                    struc=[int(i) for i in var[-3:]]
                    self.NNstrucselect.append([ncase,batchnum,loss,struc])
            print(self.NNstrucselect)
        return 

class Common_Parm:
    def __init__(self):
        self.Esoinn_setting=Esoinn_setting()
        self.Software_setting=Software_setting()
        self.Compute_setting=Compute_setting()
        self.System_setting=[]
        self.MD_setting=[]
        self.Neuralnetwork_setting=Neuralnetwork_setting()
        self.Dataset_setting=Dataset_setting()
        self.Train_setting=Train_setting()
        return 

GPARAMS=Common_Parm()

