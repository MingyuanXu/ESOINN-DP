import time
import numpy as np
from sys import stdout
import copy
from ..Neuralnetwork import *
from ..Comparm import GPARAMS
from ..Base import *
from ..MD import *
import pickle
import random
from  TensorMol import *
from parmed.amber import AmberParm
from parmed.amber import AmberMdcrd
from parmed.amber import Rst7
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
from .Basestruc import *
from ..Comparm import *
from .NNcal import*
from .DFTBcal import * 

class QMMM_FragSystem:
    def __init__(self,prmname='',crdname='',Strucdict={},Path='./',Inpath='./',Name="",resplist=[]):
        self.prmname=prmname
        self.crdname=crdname
        self.name=Name
        self.nchain=Strucdict['NCHAIN']
        self.chnpts=Strucdict['CHNPTS']
        self.chnpte=Strucdict['CHNPTE']
        self.catom=Strucdict['CENTER']
        self.reportcharge=resplist
        self.path=Path
        self.Inpath=Inpath
        self.qmcutoff=GPARAMS.Compute_setting.Qmradius
        if GPARAMS.Esoinn_setting.Loadrespnet ==True:
            self.ifresp=True 
        else:
            self.ifresp=False
        self.QMparmdict={}
        self.QMstatedict={}
        self.QMsysdict={}
        self.record_err_step=0
        self.step=0
        self.Theroylevel=GPARAMS.Compute_setting.Theroylevel 
        self.Path=Path 
        if not os.path.exists(self.Path):
            os.system("mkdir "+self.Path)
        if not os.path.exists(self.Inpath):
            os.system("mkdir -p "+self.Inpath)
        self.Get_prmtop_info()
        if self.crdname!='': 
            self.Get_init_coord()
        self.recorderr=10
        self.FullMMparm=copy.deepcopy(self.prmtop)
        mmsys=self.FullMMparm.createSystem(nonbondedMethod=NoCutoff,rigidWater=False)
        integrator=LangevinIntegrator(300*kelvin, 1/picosecond, 0.00001*picoseconds)
        self.MM_simulation=Simulation(self.FullMMparm.topology,mmsys,integrator) 
         
    def Get_init_coord(self):
        self.coords=coords_from_rst7_AMBER(self.crdname,self.natom)     
         
    def Get_prmtop_info(self):
        self.prmtop=AmberParm(self.prmname)
        self.resptop =copy.deepcopy(self.prmtop)
        natom=len(self.prmtop.atoms)
        nres=len(self.prmtop.residues)
        atomname=self.prmtop.parm_data['ATOM_NAME']
        eleindex=self.prmtop.parm_data['ATOMIC_NUMBER']
        self.atoms=eleindex 
        respt=np.array(self.prmtop.parm_data['RESIDUE_POINTER'])-1
        respts=respt
        respte=self.prmtop.parm_data['RESIDUE_POINTER'][1:]
        respte.append(natom+1)
        respte=np.array(respte)-2
        atomcrg=np.array(self.prmtop.parm_data['CHARGE'])
        if self.ifresp==True:
            self.RESPCHARGE=copy.deepcopy(atomcrg)
        rescrg=np.zeros(nres,dtype=float)
        for i in range(nres):
            rescrg[i]=round(np.sum(atomcrg[respts[i]:respte[i]+1]))
        residue=self.prmtop.parm_data['RESIDUE_LABEL']
        restype=['L' for i in range(nres)]
        resindex=[]
        resname=[]
        upt=[]
        ucharge=[]
        utype=[]
        ucutbond=[]
        ucuttype=[]
        selectC=np.zeros(nres,dtype=int)
        selectCA=np.zeros(nres,dtype=int)
        selectN=np.zeros(nres,dtype=int)
        selectCB=np.zeros(nres,dtype=int)
        for i in range(nres):
            for j in range(respts[i],respte[i]+1):
                resname.append(residue[i])
                resindex.append(i)
                if atomname[j]=='C':
                    selectC[i]=j
                if atomname[j]=='CA':
                    selectCA[i]=j
                if atomname[j]=='CB':
                    selectCB[i]=j
                if atomname[j]=='N':
                    selectN[i]=j
        for i in range(self.nchain):
            for j in range(self.chnpts[i],self.chnpte[i]+1):
                restype[j]='P'
            if residue[self.chnpts[i]]=='ACE':
                ucutbond.append([])
                ucuttype.append([])
                tmpupt=np.zeros(2,dtype=int)
                tmpupt[0]=respts[self.chnpts[i]]
                if (residue[self.chnpts[i]+1]=='PRO'):
                    tmpupt[1]=selectC[self.chnpts[i]+1]-1
                    utype.append('ORPH')
                    ucharge.append(0)
                    ucutbond[-1].append([selectCA[self.chnpts[i]+1],selectC[self.chnpts[i]+1]])
                    ucuttype[-1].append('CC')
                else:
                    tmpupt[1]=selectCA[self.chnpts[i]+1]-1
                    utype.append('ECA')
                    ucharge.append(0)
                    ucutbond[-1].append([selectN[self.chnpts[i]+1],selectCA[self.chnpts[i]+1]])
                    ucuttype[-1].append('NC')
                upt.append(tmpupt)
                Xstart=tmpupt[1]+1
                Xend=selectC[self.chnpte[i]-1]-1
                for j in range(Xstart,Xend):
                    if atomname[j]=='CA' and resname[j]!='PRO': 
                        tmpupt=np.zeros(2,dtype=int)
                        tmpupt[0]=selectCA[resindex[j]]
                        tmpupt[1]=selectC[resindex[j]]-1
                        upt.append(tmpupt)
                        strname=resname[j][::-1]
                        utype.append(strname)
                        ucharge.append(rescrg[resindex[j]])
                        ucutbond.append([])
                        ucuttype.append([])
                        ucutbond[-1].append([selectCA[resindex[j]],selectN[resindex[j]]])
                        ucuttype[-1].append('CN')
                        ucutbond[-1].append([selectCA[resindex[j]],selectC[resindex[j]]])
                        ucuttype[-1].append('CC')
                    elif (atomname[j]=='C'):
                        if residue[resindex[j]+1]=='PRO':
                            tmpupt=np.zeros(2,dtype=int)
                            tmpupt[0]=j
                            tmpupt[1]=selectC[resindex[j]+1]-1
                            upt.append(tmpupt)
                            utype.append('ORP')
                            ucharge.append(0)
                            ucutbond.append([])
                            ucuttype.append([])
                            ucutbond[-1].append([j,selectCA[resindex[j]]])
                            ucuttype[-1].append('CC')
                            ucutbond[-1].append([selectCA[resindex[j]+1],selectC[resindex[j]+1]])
                            ucuttype[-1].append('CC')
                        else:
                            tmpupt=np.zeros(2,dtype=int)
                            tmpupt[0]=j
                            tmpupt[1]=selectCA[resindex[j]+1]-1
                            upt.append(tmpupt)
                            utype.append('NCB')
                            ucharge.append(0)
                            ucutbond.append([])
                            ucuttype.append([])
                            ucutbond[-1].append([j,selectCA[resindex[j]]])
                            ucuttype[-1].append('CC')
                            ucutbond[-1].append([selectN[resindex[j]+1],selectCA[resindex[j]+1]])
                            ucuttype[-1].append('NC')
            else:
                tmpupt=np.zeros(2,dtype=int)       
                tmpupt[0]=respts[self.chnpts[i]]    
                tmpupt[1]=selectC[self.chnpts[i]]-1
                upt.append(tmpupt)
                utype.append('H')
                ucharge.append(rescrg[self.chnpts[i]])
                ucutbond.append([])
                ucuttype.append([])
                ucutbond[-1].append([selectCA[self.chnpts[i]],selectC[self.chnpts[i]]])
                ucuttype[-1].append('CC')
                Xstart=tmpupt[1]+1      
                if residue[self.chnpte[i]]=='PRO':
                    Xend=selectC[self.chnpte[i]-1]
                else:
                    Xend=selectCA[self.chnpte[i]]-1
                for j in range(Xstart,Xend):
                    if atomname[j]=='CA' and resname[j]!='PRO':
                        tmpupt=np.zeros(2,dtype=int)
                        tmpupt[0]=j
                        tmpupt[1]=selectC[resindex[j]]-1
                        utype.append('R')
                        ucharge.append(rescrg[resindex[j]])
                        upt.append(tmpupt)
                        ucutbond.append([])
                        ucuttype.append([])
                        ucutbond[-1].append([j,selectN[resindex[j]]])
                        ucuttype[-1].append('CN')
                        ucutbond[-1].append([j,selectC[resindex[j]]])
                        ucuttype[-1].append('CC')
                    elif atomname[j]=='C':
                        if residue[resindex[j]+1]=='PRO':
                            tmpupt=np.zeros(2,dtype=int)
                            tmpupt[0]=j
                            tmpupt[1]=selectC[resindex[j]+1]-1
                            upt.append(tmpupt)
                            ucharge.append(rescrg[resindex[j]+1])
                            utype.append('S')
                            ucutbond.append([])
                            ucuttype.append([])
                            ucutbond[-1].append([j,selectCA[resindex[j]]])
                            ucuttype[-1].append('CC')
                            ucutbond[-1].append([selectCA[resindex[j]+1],selectC[resindex[j]+1]])
                            ucuttype[-1].append('CC')    
                        if residue[resindex[j]+1]!='PRO':
                            tmpupt=np.zeros(2,dtype=int)
                            tmpupt[0]=j
                            tmpupt[1]=selectCA[resindex[j]+1]-1
                            upt.append(tmpupt)
                            ucharge.append(0)
                            utype.append('B')
                            ucutbond.append([])
                            ucuttype.append([])
                            ucutbond[-1].append([j,selectCA[resindex[j]]])
                            ucuttype[-1].append('CC')
                            ucutbond[-1].append([selectN[resindex[j]+1],selectCA[resindex[j]+1]])
                            ucuttype[-1].append('NC')
                tmpupt=np.zeros(2,dtype=int)
                tmpupt[0]=Xend+1
                tmpupt[1]=respte[self.chnpte[i]] 
                ucharge.append(rescrg[self.chnpte[i]])
                upt.append(tmpupt)
                utype.append('E')     
                ucutbond.append([])  
                ucuttype.append([])
                if residue[self.chnpte[i]]=='PRO':
                    ucutbond[-1].append([selectC[self.chnpte[i]-1],selectCA[self.chnpte[i]-1]])
                    ucuttype[-1].append('CC')
                else:
                    ucutbond[-1].append([selectCA[self.chnpte[i]],selectN[self.chnpte[i]]])
                    ucuttype[-1].append('CN')

        for i in range(self.chnpte[-1],nres):
            tmpupt=np.zeros(2,dtype=int)
            tmpupt[0]=respts[i]
            tmpupt[1]=respte[i]
            upt.append(tmpupt)
            ucutbond.append([])
            ucuttype.append([])
            if residue[i]=='WAT':
                restype[i]='W'
                utype.append('WAT')
            elif i==resindex[self.catom]:
                restype[i]='C'
                utype.append(residue[i])
            else:
                utype.append(residue[i])
            ucharge.append(rescrg[i])
        unum=len(upt)
        self.nres=nres;self.natom=natom
        self.respts=respts;self.respte=respte
        self.atomlist=[]
        self.fraglist=[]
        unum=unum+1
        for i in range(natom):
            a=atom(eleindex[i],atomname[i],i,resname[i],resindex[i])
            self.atomlist.append(a)
        self.nfrag=0
        self.fragindex=np.zeros(natom,dtype=int)
        for i in range(len(upt)):
            self.fragindex[upt[i][0]:upt[i][1]+1]=self.nfrag
            f=frag(self.atomlist[upt[i][0]:upt[i][1]+1],cbnd=ucutbond[i],gtype=ucuttype[i],\
                fragtype=utype[i])
            self.fraglist.append(f)
            self.nfrag=self.nfrag+1
        self.fragcharge=ucharge
        return 
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

    def Create_QMarea(self):
        self.TCRvec=self.Distance_Matrix[self.catom]
        tmpindex=np.where(self.TCRvec<self.qmcutoff)[0]
        QMfraglist=sorted(list(set([self.fragindex[m] for m in tmpindex])))
        str=''
        glist=[]
        QMidlist=[]
        QMlist=[]
        for i in QMfraglist:
            ifrag=self.fraglist[i]
            QMidlist+=list(ifrag.indexlist)
            QMlist+=ifrag.atomlist
        for i in QMfraglist:
            ifrag=self.fraglist[i]
            cbndo=ifrag.cbpto
            for j in range(len(cbndo)):
                if cbndo[j] not in QMidlist:
                    ifrag.glist[j].get_crd(self.coords[ifrag.glist[j].bpt],self.coords[ifrag.glist[j].realpt])
                    glist.append(ifrag.glist[j])
        QMlist_withg=QMlist+glist
        def get_aindex(elem):
            return elem.aindex
        QMlist_withg.sort(key=get_aindex)
        alist=np.array([m.atomnum for m in QMlist_withg])
        positions=np.array([m.crd for m in QMlist_withg])
        QMcharge=np.sum(np.array(self.fragcharge)[QMfraglist])
        QMMol=Molnew(alist,positions,QMcharge)
        self.step+=1
        return QMMol
        
    def Cal_EFQ(self):
        self.TCRvec=self.Distance_Matrix[self.catom]
        tmpindex=np.where(self.TCRvec<self.qmcutoff)[0]
        QMfraglist=sorted(list(set([self.fragindex[m] for m in tmpindex])))
        str=''
        glist=[]
        QMidlist=[]
        QMlist=[]
        for i in QMfraglist:
            ifrag=self.fraglist[i]
            QMidlist+=list(ifrag.indexlist)
            QMlist+=ifrag.atomlist
        for i in QMfraglist:
            ifrag=self.fraglist[i]
            cbndo=ifrag.cbpto
            for j in range(len(cbndo)):
                if cbndo[j] not in QMidlist:
                    ifrag.glist[j].get_crd(self.coords[ifrag.glist[j].bpt],self.coords[ifrag.glist[j].realpt])
                    glist.append(ifrag.glist[j])
        QMlist_withg=QMlist+glist
        def get_aindex(elem):
            return elem.aindex
        QMlist_withg.sort(key=get_aindex)
        self.force=np.zeros((self.natom,3))
        self.energy=0
        self.charge=np.zeros(self.natom)
        EGCMlist=[]
        alist=np.array([m.atomnum for m in QMlist_withg])
        positions=np.array([m.crd for m in QMlist_withg])
        QMcharge=np.sum(np.array(self.fragcharge)[QMfraglist])
        QMMol=Molnew(alist,positions,QMcharge)

        try:
            EGCM=(QMMol.Cal_EGCM()-GPARAMS.Esoinn_setting.scalemin)/(GPARAMS.Esoinn_setting.scalemax-GPARAMS.Esoinn_setting.scalemin)
            EGCM[ ~ np.isfinite( EGCM )] = 0
            EGCMlist.append(EGCM)
            #QMMol.belongto=GPARAMS.Esoinn_setting.Model.find_closest_cluster(GPARAMS.Train_setting.Modelnumperpoint,EGCM)
            if GPARAMS.Esoinn_setting.Model.class_id<GPARAMS.Train_setting.Modelnumperpoint:
                QMMol.belongto=[i for i in range(GPARAMS.Train_setting.Modelnumperpoint)]
            else:
                QMMol.belongto=GPARAMS.Esoinn_setting.Model.find_closest_cluster(min(GPARAMS.Train_setting.Modelnumperpoint,GPARAMS.Esoinn_setting.Model.class_id),EGCM)
        except:
            EGCM=QMMol.Cal_EGCM()
            EGCMlist.append(EGCM)
        QMMol.properties['clabel']=QMcharge
        QMSet=MSet()
        QMSet.mols.append(QMMol)
        QMSet.mols[-1].name="%s_stage_%d_MDStep_%d_%d"%(self.name,GPARAMS.Train_setting.Trainstage,self.step,len(QMSet.mols))
        if self.Theroylevel=='NN':
            try:
                NN_predict,ERROR_mols,AVG_ERR,ERROR_str,self.stepmethod=\
                    Cal_NN_EFQ(QMSet,inpath=self.Inpath)
            except:
                NN_predict,ERROR_mols,AVG_ERR,ERROR_str,self.stepmethod=\
                    Cal_Gaussian_EFQ(QMSet,self.Inpath,GPARAMS.Compute_setting.Gaussiankeywords,GPARAMS.Compute_setting.Ncoresperthreads)

            
        if self.Theroylevel=='DFTB3':
            NN_predict,ERROR_mols,AVG_ERR,ERROR_str,self.stepmethod=\
                    Cal_DFTB_EFQ(QMSet,\
                                 GPARAMS.Software_setting.Dftbparapath,\
                                 inpath=self.Inpath)

        if self.Theroylevel=="Semiqm" or self.Theroylevel=="DFT":
            NN_predict,ERROR_mols,AVG_ERR,ERROR_str,self.stepmethod=\
                    Cal_Gaussian_EFQ(QMSet,self.Inpath,GPARAMS.Compute_setting.Gaussiankeywords,GPARAMS.Compute_setting.Ncoresperthreads)

        if self.stepmethod=="NN":
            if len(ERROR_mols)>0 and self.step-self.record_err_step>5:
                self.record_err_step=self.step    
            else:
                self.ERROR_mols=[]
                self.EGCMlist=[]
            
        self.recorderr=AVG_ERR
        self.QMarea_QMforce=NN_predict[0][1]
        self.QMarea_QMenergy=NN_predict[0][0]
        if self.ifresp==True:
            self.QMarea_respcharge=NN_predict[0][3] 

        if self.ifresp==True:
            self.RESPCHARGE=copy.deepcopy(np.array(self.prmtop.parm_data['CHARGE']))
            if len(self.QMarea_respcharge)!=0:
                for i in range(len(QMlist_withg)):
                    if QMlist_withg[i].bpt!=-1:
                        self.RESPCHARGE[QMlist_withg[i].bpt]=0
                    else:
                        self.RESPCHARGE[QMlist_withg[i].realpt]=0

        for i in range(len(QMlist_withg)):
            if QMlist_withg[i].bpt!=-1:
                self.force[QMlist_withg[i].bpt] +=self.QMarea_QMforce[i]
                if self.ifresp==True:
                    if len(self.QMarea_respcharge)!=0:
                        self.RESPCHARGE[QMlist_withg[i].bpt]+=self.QMarea_respcharge[i]
            else:
                self.force[QMlist_withg[i].realpt]+=self.QMarea_QMforce[i]
                if self.ifresp==True:
                    if len(self.QMarea_respcharge)!=0:
                        self.RESPCHARGE[QMlist_withg[i].realpt]+=self.QMarea_respcharge[i]

        self.energy+=self.QMarea_QMenergy
        chargestr=''
        if self.ifresp==True:
            chargestr="Step: %d RESP CHARGE OF MBG: "%self.step
            for i in self.reportcharge: 
                chargestr+="    %f  "%self.RESPCHARGE[i]
        chargestr+='\n'
        qmsysname=''
        for i in QMlist:
            qmsysname+=i.aname
        if self.ifresp==True:
            if self.step%100==0 :
                self.resptop.parm_data['CHARGE']=self.RESPCHARGE
                self.QMparmdict={}
                self.QMsysdict={}
        if qmsysname not in self.QMparmdict.keys():
            self.QMparm=copy.deepcopy(self.resptop)
            maskstr='!@'
            for i in QMlist:
                maskstr+='%d,'%(i.realpt+1)
            maskstr=maskstr[:-1]
            self.QMparm.strip(maskstr)
            self.QMparmdict[qmsysname]=self.QMparm
        else:
            self.QMparm=self.QMparmdict[qmsysname]
        if qmsysname not in self.QMsysdict.keys():
            tmpparm=copy.deepcopy(self.QMparm)
            qmsys=tmpparm.createSystem(nonbondedMethod=NoCutoff,rigidWater=False)
            integrator=LangevinIntegrator(300*kelvin, 1/picosecond, 0.00001*picoseconds)
            self.QMarea_simulation=Simulation(tmpparm.topology,qmsys,integrator) 
            positions=np.array([m.crd for m in QMlist])
            self.QMarea_simulation.context.setPositions(positions/10)
            self.QMsysdict[qmsysname]=self.QMarea_simulation
        else:
            self.QMarea_simulation=self.QMsysdict[qmsysname]
        positions=np.array([m.crd for m in QMlist])
        self.QMarea_simulation.context.setPositions(positions/10)
        self.QMarea_state=self.QMarea_simulation.context.getState(getEnergy=True,getForces=True,getPositions=True)
        self.QMarea_MMenergy=self.QMarea_state.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
        self.QMarea_MMforce=self.QMarea_state.getForces(asNumpy=True).value_in_unit(kilocalories/(angstrom*mole))
        for i in range(len(QMlist)):
            self.force[QMlist[i].realpt]-=self.QMarea_MMforce[i]
        self.energy-=self.QMarea_MMenergy
        if self.ifresp==True:
            if self.step%100==0:
                self.FullMMparm=copy.deepcopy(self.prmtop)
                self.FullMMparm.parm_data['CHARGE']=self.RESPCHARGE
                mmsys=self.FullMMparm.createSystem(nonbondedMethod=NoCutoff,rigidWater=False)
                integrator=LangevinIntegrator(300*kelvin, 1/picosecond, 0.00001*picoseconds)
                self.MM_simulation=Simulation(self.FullMMparm.topology,mmsys,integrator) 
                positions=np.array(self.coords)
                self.MM_simulation.context.setPositions(positions/10)
                self.MMstate=self.MM_simulation.context.getState(getEnergy=True,getForces=True,getPositions=True)
            
        positions=np.array(self.coords)
        self.MM_simulation.context.setPositions(positions/10)
        self.MMstate=self.MM_simulation.context.getState(getEnergy=True,getForces=True,getPositions=True)
        self.FullMM_energy=self.MMstate.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
        self.FullMM_force=self.MMstate.getForces(asNumpy=True).value_in_unit(kilocalories/(angstrom*mole))
        self.force+=self.FullMM_force
        self.energy+=self.FullMM_energy

        #num=0
        #if self.stepmethod=='DFTB':
        #    for i in range(len(QMlist_withg)):
        #        if QMlist_withg[i].bpt==-1:
        #            realpt=QMlist_withg[i].realpt
        #            print (realpt,QMlist_withg[i].aname,self.coords[realpt],self.force[realpt],self.FullMM_force[realpt],self.QMarea_MMforce[num],self.QMarea_QMforce[i])
        #            num+=1
        self.step+=1
        return self.force/627.51*JOULEPERHARTREE,self.energy/627.51,AVG_ERR,ERROR_mols,EGCMlist,chargestr 

    def update_crd(self):
        for i in range(self.natom):
            self.atomlist[i].get_crd(self.coords[i])    

