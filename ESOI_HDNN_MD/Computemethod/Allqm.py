import time
import numpy as np
from  TensorMol import *
import pickle
import random

from sys import stdout
import copy
from ..Comparm import * 
from ..Base import * 

class FullQM_System:
    def __init__(self,atoms,coords,totalcharge=0,Path='./',Inpath='./'):
        self.atoms=atoms
        self.coords=coords
        self.natom=len(atoms)
        self.qmcutoff=GPARAMS.Compute_setting.Qmradius
        self.Theroylevel=GPARAMS.Compute_setting.Theroylevel
        self.Path=Path
        if not os.path.exists(self.Path):
            os.system("mkdir %s"%self.Path)
        self.Inpath=Inpath
        if not os.path.exists(self.Inpath):
            os.system("mkdir %s"%self.Inpath)
        self.step=0
        self.totalcharge=totalcharge 
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
            EGCM=(QMMol.Cal_EGCM()-GPARAMS.Esoinn_setting.scalemin)/(GPARAMS.Esoinn_setting.scalemax-Esoinn_setting.scalemin)
            EGCM[ ~ np.isfinite( EGCM )] = 0
            EGCMlist.append(EGCM)
            QMMol.belongto=GPARAMS.Esoinn_setting.Model.find_closest_cluster(min(GPARAMS.Train_setting.Modelnumperpoint,GPARAMS.Esoinn_setting.Model.class_id),EGCM)
        except:
            EGCM=QMMol.Cal_EGCM()
            EGCMlist.append(EGCM)
        QMMol.properties['clabel']=self.totalcharge 
        QMSet=MSet()
        QMSet.mols.append(QMMol)
        QMSet.mols[-1].name="Stage_%d_MDStep_%d_%d"%(GPARAMS.Train_setting.Trainstage,self.step,len(QMSet.mols))
        if self.Theroylevel=="NN":
            NN_predict,ERROR_mols,AVG_ERR,ERROR_str,self.stepmethod=\
                    Cal_NN_EFQ(QMSet,inpath=self.Inpath)
        elif self.Theroylevel=="DFTB3":
            NN_predict,ERROR_mols,AVG_ERR,ERROR_str,self.stepmethod=\
                    Cal_DFTB_EFQ(QMSet,\
                                GPARAMS.Software_setting.Dftbparapath,\
                                inpath=self.Inpath)
        elif self.Theroylevel=="Semiqm":
            NN_predict,ERROR_mols,AVG_ERR,ERROR_str,self.stepmethod=\
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
        if len(ERROR_mols)>0 and self.step-self.err_step>5:
            self.err_step=self.step
        else:
            ERROR_mols=[]
        self.recorderr=AVG_ERR
        self.ERROR_file.write('Step: %d  '%self.step)
        self.ERROR_file.write(ERROR_str) 
        self.force=NN_predict[0][1]
        self.energy=NN_predict[0][0]
        return self.force/627.51*JOULEPERHARTREE,self.energy/627.51,AVG_ERR,ERROR_mols,EGCMlist 
    def update_crd(self):
        pass

