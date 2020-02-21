from __future__ import absolute_import
from __future__ import print_function
import sys
sys.path.append('../Esoinn')
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
from parmed.amber import AmberParm
from parmed.amber import AmberMdcrd
from parmed.amber import Rst7
import random
import pickle
from multiprocessing import Queue,Process 
from .Thermostat import * 
from .analysis import * 
from ..Comparm import *

class Simulation():
    def __init__(self,sys,MD_setting):
        self.name=MD_setting.Name 
        self.sys=sys
        self.path='./'+MD_setting.Name+'/' 
        self.format=MD_setting.Mdformat
        self.T=MD_setting.Temp
        self.maxstep=MD_setting.Mdmaxsteps
        self.dt=MD_setting.Mddt
        self.icap=MD_setting.Icap
        self.ibox=MD_setting.Ibox
        self.box=MD_setting.Box 
        self.center=MD_setting.Center 
        self.radius=MD_setting.Capradius 
        self.fcap=MD_setting.Capf
        self.MODE=MD_setting.Mode 
        self.MDThermostat=MD_setting.Thermostat 
        self.MDV0=MD_setting.Mdv0 
        self.stageindex=MD_setting.Stageindex
        self.Outfile=open(self.path+MD_setting.Name+'_%d.mdout'%self.stageindex,'w')
        self.Respfile=open(self.path+MD_setting.Name+'_%d.resp'%self.stageindex,'w')
        self.Nprint=MD_setting.Nprint
        self.Maxerr=GPARAMS.Neuralnetwork_setting.Maxerr 
        self.Miderr=GPARAMS.Neuralnetwork_setting.Miderr 
        self.Midrate=GPARAMS.Neuralnetwork_setting.Midrate 
        return

    def MD(self,QMQueue=None):
        self.sys.Create_DisMap()
        self.sys.Update_DisMap()
        self.sys.update_crd()
        f,e,AVG_ERR,ERROR_mols,EGCMlist,chargestr=self.sys.Cal_EFQ()
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
        if self.format=="Amber":
            self.trajectory=AmberMdcrd(self.path+self.name+'_%d.mdcrd'%self.stageindex,natom=self.sys.natom,hasbox=False,mode='w')
            self.restart=Rst7(natom=len(self.atoms))
            self.trajectory.add_coordinates(self.x)

        if self.MDV0=="Random":
            np.random.seed()
            self.v=np.random.randn(*self.x.shape)
            Tstat = Thermostat(self.m, self.v,self.T,self.dt)
        elif self.MDV0=="Thermal":
            print ("+++++++++++++++++self.T++++++++++%d"%self.T)
            self.v = np.random.normal(size=self.x.shape) * np.sqrt(1.38064852e-23 * self.T / self.m)[:,None]

        self.Tstat = None
        if (self.MDThermostat=="Rescaling"):
            self.Tstat = Thermo(self.m,self.v,self.T,self.dt)
        elif (self.MDThermostat=="Andersen"):
            self.Tstat = Andersen(self.m,self.v,self.T,self.dt)

        self.a=pow(10.0,-10.0)*np.einsum("ax,a->ax", self.f, 1.0/self.m)
        if self.format=="Amber":
            self.restart.coordinates=self.x
            self.restart.vels=self.v
        
        step=0
        self.md_log=np.zeros((self.maxstep+1,7))
        res_order=np.array(range(1,self.sys.nres))
        ERROR=0
        ERROR_record=[]
        method_record=0
        Temp_record=[]
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
            self.sys.Update_DisMap()
            self.sys.update_crd()
            f,EPot,ERROR,ERROR_mols,EGCMlist,chargestr=self.sys.Cal_EFQ()
            ERROR_record.append(ERROR)
            if self.sys.stepmethod=='Gaussian':
                method_record+=1
            
            if ERROR>self.Miderr and ERROR<self.Maxerr:
                miderr_num+=1
            if self.MODE=='Train':
                if QMQueue!=None:
                    if ERROR>self.Maxerr:
                        QMQueue.put(ERROR_mols)
                    elif miderr_num<self.maxstep*self.Midrate:
                        QMQueue.put(ERROR_mols)
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
            Temp_record.append(Teff)
        
            if (step%10==0 ):
                if self.format=="Amber":
                    self.trajectory.add_coordinates(self.x)
            step+=1
            AVG_ERR=np.mean(np.array(ERROR_record[-1:-30]))
            AVG_TEMP=np.mean(np.array(Temp_record[-1:-30]))
            if AVG_ERR>GPARAMS.Train_setting.rmse**2*GPARAMS.Train_setting.Modelnumperpoint*4:
                MD_Flag=False
            if method_record>10:
                MD_Flag=False 
            if AVG_TEMP>350:
                MD_Flag=False
            if (step%self.Nprint==0 ):
                if self.format=="Amber":
                    if MD_Flag:
                        self.restart.coordinates=self.x
                        self.restart.vels=self.v
                        self.restart.write(self.path+self.name+'_%d.rst7'%self.stageindex)
                        self.steprecord=step
                    else:
                        file=open('%strajd%.trajin'%(self.path,self.stageindex),'w')
                        file.write('trajin %s %d %d 1\n'%(self.name+'_%d.mdcrd'%self.stageindex,0,math.ceil(self.steprecord,10)))
                        file.write('trajout %s\n' %(self.name+'_%d.mdcrd'%self.stageindex))
                        os.system("cd %s && cpptraj -p %s < traj%d.trajin > traj%d.out && cd .."%(self.path,self.name+'.prmtop',self.stageindex,self.stageindex))
            if MD_Flag==True:
                self.Outfile.write("%s Step: %i time: %.1f(fs) KE(kJ): %.5f PotE(Eh): %.5f ETot(kJ/mol): %.5f Teff(K): %.5f MAX ERROR: %.3f Method: %s  AVG_ERR: %f AVG_TEMP: %f \n"\
                                   %(self.name, step, self.t, self.KE*len(self.m)/1000.0, self.EPot, self.KE*len(self.m)/1000.0+(self.EPot)*KJPERHARTREE, Teff,ERROR,self.sys.stepmethod,AVG_ERR,AVG_TEMP))
                self.Outfile.flush()
                self.Respfile.write(chargestr)
                self.Respfile.write(chargestr)
            else:
                self.Outfile.write("AVG ERR: %.3f , MD will stop~~!!"%AVG_ERR)
                self.Outfile.flush()
        self.Outfile.close()
        self.Respfile.close()
        return 

