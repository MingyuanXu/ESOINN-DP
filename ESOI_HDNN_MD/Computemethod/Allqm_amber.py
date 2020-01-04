import time
import numpy as np
from  TensorMol import *
import pickle
import random
from ..Neuralnetwork import *
from ..Comparm import GPARAMS
from ..Base import *
from ..MD import *

from sys import stdout
import copy
from parmed.amber import AmberParm
from parmed.amber import AmberMdcrd
from parmed.amber import Rst7
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
from .NNcal import*
from .DFTBcal import * 
from .Basestruc import *
from ..Comparm import *

class FullQM_System_Amber:
    def __init__(self,prmname='',crdname='',Path='./',Inpath='./',Name=""):
        self.prmname=prmname
        self.crdname=crdname 
        self.name=Name
        self.qmcutoff=GPARAMS.Compute_setting.Qmradius
        self.Theroylevel=GPARAMS.Compute_setting.Theroylevel
        self.Path=Path
        if not os.path.exists(self.Path):
            os.system("mkdir %s"%self.Path)
        self.Inpath=Inpath
        if not os.path.exists(self.Inpath):
            os.system("mkdir %s"%self.Inpath)
        self.step=0
        self.err_step=0
        self.Get_prmtop_info()
        if self.crdname!='':
            self.Get_init_coord()
        self.FullMMparm=copy.deepcopy(self.prmtop)
        mmsys=self.FullMMparm.createSystem(nonbondedMethod=NoCutoff,rigidWater=False)
        integrator=LangevinIntegrator(300*kelvin,1/picosecond,0.00001*picoseconds)
        self.MMsimulation=Simulation(self.FullMMparm.topology,mmsys,integrator)
    def Get_init_coord(self):
        self.coords=coords_from_rst7_AMBER(self.crdname,self.natom)
    def Get_prmtop_info(self):
        self.prmtop=AmberParm(self.prmname)
        self.natom=len(self.prmtop.atoms)
        self.nres=len(self.prmtop.residues)
        atomname=self.prmtop.parm_data["ATOM_NAME"]
        eleindex=self.prmtop.parm_data['ATOMIC_NUMBER']
        atomcrg=self.prmtop.parm_data['CHARGE']
        self.totalcharge=round(np.sum(atomcrg))
        self.atoms=np.array(eleindex)
        self.respts=np.array(self.prmtop.parm_data['RESIDUE_POINTER'])-1
        respte=self.prmtop.parm_data['RESIDUE_POINTER'][1:]
        respte.append(self.natom+1)
        self.respte=np.array(respte)-2
        self.rescrg=np.zeros(self.nres) 
        for i in range(self.nres):
            self.rescrg[i]=round(np.sum(atomcrg[self.respts[i]:self.respte[i]+1]))
        print (self.rescrg)
    def Create_DisMap(self):
        d1=np.zeros((self.natom,self.natom),dtype=float)
        np.fill_diagonal(d1,0.00000000001)
        self.Dgraph=tf.Graph()
        with self.Dgraph.as_default():
            self.tfcrd=tf.placeholder(shape=[self.natom,3],dtype=tf.float64,name='coordinate')
            self.tfR=tf.reshape(tf.reduce_sum(self.tfcrd*self.tfcrd,1),[-1,1])
            self.tfDM=tf.sqrt(self.tfR-2*tf.matmul(self.tfcrd,tf.transpose(self.tfcrd))\
                    +tf.transpose(self.tfR)+tf.constant(d1))
        self.Dsess=tf.Session(graph=self.Dgraph)
        
    def Update_DisMap(self):
        self.Distance_Matrix=self.Dsess.run(self.tfDM,{self.tfcrd:self.coords})
        return 

    def Cal_EFQ(self):
        self.step+=1
        self.force=np.zeros((self.natom,3))
        self.energy=0
        self.charge=np.zeros(self.natom)
        EGCMlist=[]
        QMMol=Molnew(self.atoms,self.coords,self.totalcharge)
        self.QMMol=QMMol
        try:
        #if True:
            EGCM=(QMMol.Cal_EGCM()-GPARAMS.Esoinn_setting.scalemin)/(GPARAMS.Esoinn_setting.scalemax-GPARAMS.Esoinn_setting.scalemin)
            EGCM[ ~ np.isfinite( EGCM )] = 0
            EGCMlist.append(EGCM)
            #QMMol.belongto=self.ESOINN_MODEL.find_closest_cluster(GPARAMS.Train_setting.Modelnumperpoint,EGCM)
            if GPARAMS.Esoinn_setting.Model.class_id<GPARAMS.Train_setting.Modelnumperpoint:
                QMMol.belongto=[i for i in range(GPARAMS.Train_setting.Modelnumperpoint)]
            else:
                QMMol.belongto=GPARAMS.Esoinn_setting.Model.find_closest_cluster(min(GPARAMS.Train_setting.Modelnumperpoint,GPARAMS.Esoinn_setting.Model.class_id),EGCM)
        except:
            EGCM=QMMol.Cal_EGCM()
            EGCMlist.append(EGCM)
        QMMol.properties['clabel']=self.totalcharge 
        QMSet=MSet()
        QMSet.mols.append(QMMol)
        QMSet.mols[-1].name="%s_Stage_%d_MDStep_%d_%d"%(self.name,GPARAMS.Train_setting.Trainstage,self.step,len(QMSet.mols))
        ERROR_mols=[]
        if self.Theroylevel=="NN":
            Predict,ERROR_mols,AVG_ERR,ERROR_str,self.stepmethod=\
                    Cal_NN_EFQ(QMSet,inpath=self.Inpath)
        elif  self.Theroylevel=="DFTB3":
            Predict,ERROR_mols,AVG_ERR,ERROR_str,self.stepmethod=\
                    Cal_DFTB_EFQ(QMSet,\
                                GPARAMS.Software_setting.Dftbparapath,\
                                inpath=self.Inpath)
        elif self.Theroylevel=="Semiqm":
            Predict,ERROR_mols,AVG_ERR,ERROR_str,self.stepmethod=\
                    Cal_Gaussian_EFQ(QMSet,self.Inpath,\
                                     GPARAMS.Compute_setting.Gaussiankeywords,\
                                     GPARAMS.Compute_setting.Ncoresperthreads)
        """
        if self.level=='REACX':
            if self.step==0:
                NN_predict,ERROR_mols,AVG_ERR,ERROR_str,self.stepmethod=Cal_REACX_EFQ(QMSet,parapath=self.parapath,inpath=self.filedir,onlydata=False)
            else:
                NN_predict,ERROR_mols,AVG_ERR,ERROR_str,self.stepmethod=Cal_REACX_EFQ(QMSet,parapath=self.parapath,inpath=self.filedir,onlydata=True)

        """
        if self.Theroylevel=="Amber" or AVG_ERR>200:
            positions=np.array(self.coords)
            self.MMsimulation.context.setPositions(positions/10)
            self.MMstate=self.MMsimulation.context.getState(getEnergy=True,getForces=True,getPositions=True)
            self.energy=self.MMstate.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
            self.force=self.MMstate.getForces(asNumpy=True).value_in_unit(kilocalories/(angstrom*mole)) 
            self.recorderr=0
            AVG_ERR=999
            self.stepmethod="Amber"
            ERROR_str=''
            ERROR_mols.append([QMMol,999])
        else:
            self.recorderr=AVG_ERR
            self.force=Predict[0][1]
            self.energy=Predict[0][0]
        if len(ERROR_mols)>0 and self.step-self.err_step>5:
            self.err_step=self.step
        else:
            ERROR_mols=[]
            EGCMlist=[]
        return self.force/627.51*JOULEPERHARTREE,self.energy/627.51,AVG_ERR,ERROR_mols,EGCMlist,"" 
    def update_crd(self):
        pass

