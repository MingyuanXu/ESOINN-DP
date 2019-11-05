import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
from parmed.amber import AmberParm
from parmed.amber import AmberMdcrd
from parmed.amber import Rst7
import pickle
from ..Base import *
import random
from multiprocessing import Queue,Process 

class BumpHolder(ForceHolder):
    def __init__(self,CV_num,Max_bump,Height,Width):
        self.max_num=Max_bump
        self.cv_num=cv_num
        self.h_a=Height
        self.w_a=Width
        self.h=None
        self.w=None
        self.Prepare()
        return 
    def Prepare():
        self.Bumpgraph=tf.Graph()
        with self.Bumpgraph.as_default():
            self.xyzs_pl=tf.placeholder(tf.float64,shape=(cv_num,3))
            self.xyzt_pl=tf.placeholder(tf.float64,shape=(cv_num,3))
            self.dis_pl=tf.placeholder(tf.float64,shape=(None,cv_num,1))
            self.h=tf.Variable(self.h_a,dtype=tf.float64)
            self.w=tf.Variable(self.w_a,dtype=tf.float64)
            self.dis=tf.reduce_sum(tf.square(self.xyzs_pl-self.xyzt_pl),1)
            init=tf.global_variables_initializer()
        self.Bumpsess=tf.Session(graph=self.Bumpgraph,config=tf.ConfigProto(allow_soft_placement=True))
        #self.sess.run(self.dis,{self.xyzs_pl:})

class Meta_Simulation():
    def __init__(self,sys,MD_name,MDctrl,CVs):
        self.Outfile=open(MD_name+'.mdout','w')
        #MD setting:
        self.T=MDctrl["MDTemp"]
        self.maxstep=MDctrl["MDMaxStep"]
        self.dt=MDctrl["MDdt"]
        self.icap=MDctrl['icap']
        self.center=MDctrl['center']
        self.radius=MDctrl['radius']
        self.fcap=MDctrl['fcap']
        self.MDmethod=MDctrl['Method']
        self.MODE=MDctrl['MODE']
        self.MDThermostat=MDctrl["MDThermostat"]
        self.MDV0=MDctrl["MDV0"]
        self.sys=sys
        self.times=0
        self.name=MD_name
        return

    def MD(self,QMQueue=None,ESOINNQueue=None):
        self.sys.Create_DisMap()
        if self.MDmethod=='ONIOM':
            self.sys.Update_DisMap()
            self.sys.update_crd()
            #self.sys.Create_QMarea()
            #self.sys.Generate_All_MBE_term_General()
            f,e,AVG_ERR,ERROR_mols,EGCMlist=self.sys.Cal_EFQ()
        self.EPot0=e
        self.EPot=e
        self.EnergyStat=OnlineEstimator(self.EPot0) 
        self.RealPot=0.0
        self.t=0.0
        self.KE=0.0
        self.atoms=self.sys.atoms
        self.m=np.array(list(map(lambda x:ATOMICMASSES[x-1],self.atoms)))
        self.x=self.sys.coords
        self.v=np.zeros(self.x.shape)
        self.a=np.zeros(self.x.shape)
        self.f=f
        self.md_log=None
        self.trajectory=AmberMdcrd(self.name+'.mdcrd',natom=self.sys.natom,hasbox=False,mode='w')
        self.restart=Rst7(natom=len(self.atoms))
        self.trajectory.add_coordinates(self.x)
        if self.MDV0=="Random":
            np.random.seed()
            self.v=np.random.randn(*self.x.shape)
            Tstat = Thermostat(self.m, self.v)
        elif self.MDV0=="Thermal":
            self.v = np.random.normal(size=self.x.shape) * np.sqrt(1.38064852e-23 * self.T / self.m)[:,None]
        self.Tstat = None
        if (self.MDThermostat=="Rescaling"):
                self.Tstat = Thermo(self.m,self.v)
        elif (self.MDThermostat=="Andersen"):
                self.Tstat = Andersen(self.m,self.v)
        self.a=pow(10.0,-10.0)*np.einsum("ax,a->ax", self.f, 1.0/self.m)
        self.restart.coordinates=self.x
        self.restart.vels=self.v
        step=0
       
        self.md_log=np.zeros((self.maxstep+1,7))
        res_order=np.array(range(1,self.sys.nres))
        ERROR=0
        ERROR_record=[]
        miderr_num=0
        MD_Flag=True
        while step < self.maxstep and MD_Flag:
            self.t+=self.dt
            t1=time.time()
            x_new=self.x+self.v*self.dt+0.5*self.a*self.dt**2
            if self.icap==True:
                x_new=x_new-x_new[self.center]
            self.sys.coords=x_new
            f=x_new;EPot=0;ERROR_mols=[]
            if self.MDmethod=='ONIOM':
                self.sys.Update_DisMap()
                self.sys.update_crd()
            #    self.sys.Create_QMarea()
                f,EPot,ERROR,ERROR_mols,EGCMlist=self.sys.Cal_EFQ()
            ERROR_record.append(ERROR)
            if ERROR>9 and ERROR<25:
                miderr_num+=1
            if self.MODE=='TRAIN':
                if QMQueue!=None:
                    if ERROR>25:
                        QMQueue.put(ERROR_mols)
                    elif miderr_num<5000:
                        QMQueue.put(ERROR_mols)
                if ESOINNQueue!=None:
                    ESOINNQueue.put(EGCMlist)
            self.EPot=EPot
            if self.icap==True:
                Vec=(self.sys.Distance_Matrix[self.center]-self.radius)/self.radius
                for i in range(len(x_new)):
                    if Vec[i]>0:
                        tmpvec=(x_new[i]-x_new[self.center])
                        tmpvec=tmpvec/np.sqrt(np.sum(tmpvec**2))
                        f[i]=f[i]-tmpvec*self.fcap*Vec[i]*JOULEPERHARTREE/627.51

            a_new=pow(10.0,-10.0)*np.einsum("ax,a->ax", f, 1.0/self.m)
            v_new=self.v+0.5*(self.a+a_new)*self.dt
            if self.MDThermostat!=None and step%1==0:
                v_new=self.Tstat.step(self.m,v_new,self.dt)
            self.a=a_new
            self.v=v_new
            self.x=x_new
            self.f=f
            self.md_log[step,0]=self.t
            self.md_log[step,4]=self.KE
            self.md_log[step,5]=self.EPot
            self.md_log[step,6]=self.KE+(self.EPot-self.EPot0)*JOULEPERHARTREE
            avE,Evar=self.EnergyStat(self.EPot)
            self.KE= KineticEnergy(self.v,self.m)
            Teff = (2./3.)*self.KE/IDEALGASR
            if (step%10==0 ):
                self.trajectory.add_coordinates(self.x)
            if (step%10==0 ):
                self.restart.coordinates=self.x
                self.restart.vels=self.v
                self.restart.write(self.name+'.rst7')
            step+=1
            AVG_ERR=np.mean(np.array(ERROR_record[-1:-50]))
            if AVG_ERR>200:
                MD_Flag=False
            if MD_Flag==True:
                self.Outfile.write("%s Step: %i time: %.1f(fs) KE(kJ): %.5f PotE(Eh): %.5f ETot(kJ/mol): %.5f Teff(K): %.5f MAX ERROR: %.3f Method: %s\n" %(self.name, step, self.t, self.KE*len(self.m)/1000.0, self.EPot, self.KE*len(self.m)/1000.0+(self.EPot)*KJPERHARTREE, Teff,ERROR,self.sys.stepmethod))
                self.Outfile.flush()
            else:
                self.Outfile.write("AVG ERR: %.3f , MD will stop~~!!"%AVG_ERR)
                self.Outfile.flush()
        self.Outfile.close()
        return 
