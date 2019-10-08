import json
import numpy as np
from itertools import product
from TensorMol import *
class PARAMETER:
    def __init__(self):
        self.MAX_NUM=1
        self.ATYPE=['O']
        self.AMAX=[1]
        self.APT=[[0,0]]
        self.E_DICT={'ZN':30,'C':6,'O':8,'H':1,'N':7,'P':15,'MG':12,'GA':20,'HG':80,'CU':29}
        self.DICT_E={30:'ZN',6:'C',8:'O',1:'H',7:'N',15:'P',12:'MG',20:'GA',80:'HG',29:'CU'}
        self.E_INDEX=[8]
        self.TARGET='Energy'
        self.RC=10.0
        self.RC_buffer=2.0
        self.PARM7=''
    def UPDATE(self,DICT):
        if 'MAX_NUM' in DICT.keys():
            self.MAX_NUM=DICT['MAX_NUM']
        if 'ATYPE' in DICT.keys():
            self.ATYPE=DICT['ATYPE']
        if 'AMAX' in DICT.keys():
            self.AMAX=DICT['AMAX']
        if len(self.AMAX)==len(self.ATYPE):
            if self.MAX_NUM!= np.sum(self.AMAX):
                print ("ERROR Parameter init wrong: MAX_NUM != sum(AMAX) !")
                print ("ERROR: PARAMETER INIT WRONG!---- MAX_NUM != sum(AMAX) !")
            else :
                self.APT=np.zeros((len(self.ATYPE),2),dtype=int)
                for i in range(len(self.ATYPE)):
                    if i>=1:
                        self.APT[i][0]=self.APT[i-1][1]+1
                    self.APT[i][1]=self.APT[i][0]+self.AMAX[i]-1
                self.E_INDEX=np.zeros(self.MAX_NUM,dtype=int)
                for i,j in product(range(self.MAX_NUM),range(len(self.ATYPE))):
                    if i>= self.APT[j][0] and i<= self.APT[j][1]:
                        self.E_INDEX[i]=self.ATYPE[j]
        else:
            print ("ERROR: PARAMETER INIT WRONG!---- len(APT) != len(AMAX) !")
        if 'RC' in DICT.keys():
            self.RC=DICT['RC']
        if 'TARGET' in DICT.keys():
            self.TARGET=DICT['TARGET']
        if 'PARM7' in DICT.keys():
            self.PARM7=DICT['PARM7']
    def SHOW(self):
        print ('MAX_NUM: ', self.MAX_NUM)
        print ('ATYPE:   ', self.ATYPE)
        print ('AMAX:    ', self.MAX_NUM)
        print ('APT:     ', self.APT)
        print ('RC:      ', self.RC)
        print ('E_INDEX: ', self.E_INDEX)
        print ('PARM:    ', self.PARM7)

class SUBNET_CTRL:
    def __init__(self):
        self.NAME=''
        self.STRUC=[80 , 60 , 40]
        self.GNUM=0
        self.GPARA=[]
        self.HIDDEN_LAYER=[]
        self.SFACTOR=[]

import numpy as np 
import pickle
from itertools import product 
from rdkit import Chem
from global_var import *

class TRAINPOINT():
    def __init__(self,Crd,Mask,Trans,Natom,F=0,Q=0,E=0,Es=0,El=0):
        self.CRD=Crd
        self.Mask=Mask
        self.Trans=Trans
        self.Natom=Natom
        self.FORCE=F
        self.CHARGE=Q
        self.ENERGY=E
        self.ES=Es
        self.El=El
        self.CoulombMartrix=0.0
        
    def writeBlock(self):
        Block=[]
        Block.append('MODEL:\n')
        Block.append('Natom: %d Energy: %.4f %.4f %.4f\n'%(self.Natom,self.ENERGY,self.ES,self.El))
        for i in range(len(self.CRD)):
            Block.append('%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.2f %d'\
                    %(i,self.CRD[i][0],self.CRD[i][1],self.CRD[i][2],\
                    self.FORCE[i][0],self.FORCE[i][1],self.FORCE[i][1],\
                    self.CHARGE[i],self.Mask[i],self.Trans[i]))
        return 

    def cal_CoulombMartrix(self,P,dummy_water=0):
        dummy_list=[[] for m in P.ATYPE] 
        for i in range(P.MAX_NUM):
            if self.Mask[i]==0:
                for j in range(len(P.ATYPE)):
                    if P.E_INDEX[i]==P.ATYPE[j]:
                        dummy_list[j].append(i)            
        C=np.zeros((len(self.CRD),len(self.CRD)),dtype=float)
        for i in range(len(self.CRD)):
            for j in range(len(self.CRD)):
                #if i==j and self.Mask[i]!=0:
                if i==j :
                    C[i][j]=0.5*P.E_INDEX[i]**2.4
                if i!=j and self.Mask[i]!=0 and self.Mask[j]!=0:
                    R=np.sqrt(np.sum((self.CRD[i]-self.CRD[j])**2))
                    C[i][j]=P.E_INDEX[i]*P.E_INDEX[j]/R
        if dummy_water==1:
            O_index=P.ATYPE.index(8);H_index=P.ATYPE.index(1)
            if len(dummy_list[O_index])>0 and len(dummy_list[H_index])>0:
                _,dummy_water_mod=np.divmod(len(dummy_list[H_index]),len(dummy_list[O_index]))
                if dummy_water_mod==0:
                    dummy_water_num=len(dummy_list[O_index])
                    for i in range(dummy_water_num):
                        C[dummy_list[O_index][i]][dummy_list[H_index][2*i]]=8.0/0.96
                        C[dummy_list[H_index][2*i]][dummy_list[O_index][i]]=8.0/0.96
                        C[dummy_list[O_index][i]][dummy_list[H_index][2*i+1]]=8.0/0.96
                        C[dummy_list[H_index][2*i+1]][dummy_list[O_index][i]]=8.0/0.96
                        C[dummy_list[H_index][2*i+1]][dummy_list[H_index][2*i]]=1.0/1.56795
                        C[dummy_list[H_index][2*i]][dummy_list[H_index][2*i+1]]=1.0/1.56795                       
#        for i in range(len(P.ATYPE)):
#            Ci=C[P.APT[i][0]:P.APT[i][1]+1]
#            C_tmp=np.argsort(-np.sum(Ci,1))
#            Ci=Ci[C_tmp]
#            C[P.APT[i][0]:P.APT[i][1]+1]=Ci
        self.CoulombMartrix=C
        self.EGCM,self.EVEC=np.linalg.eig(self.CoulombMartrix)
        self.EGCM=np.sort(-self.EGCM)
        return self

class DATAFRAME():

    def __init__(self,DATABLOCK):
        self.MOLECULE=Chem.MolFromPDBBlock(DATABLOCK,removeHs=False)
        var=DATABLOCK.split('\n')
        pt=0;fpt=0;ept=0;dpt=0;
        for eachline in var:
            pt=pt+1
            if 'FORCE and CHARGE' in eachline:
                fpt=pt
            if 'ENERGY' in eachline:
                ept=pt
            if 'DIPOLE' in eachline:
                dpt=pt
#            if 'QUADRAPOLE' in eachline:
#                qpt=pt
        self.FORCE=[]
        self.CHARGE=[]
        self.ENERGY=0.0
        self.DIPOLE=[]
#        self.QUADRAPOLE=[]
        if fpt!=0 and ept!=0:
            for j in range(self.MOLECULE.GetNumAtoms()):
                tmp=var[fpt+j].split()
                self.FORCE.append([float(tmp[m]) for m in range(3)])
                self.CHARGE.append(float(tmp[-1]))
            self.ENERGY=float(var[ept].split()[0])    
        if dpt!=0:
            tmp=var[dpt].split()
            self.DIPOLE=[float(m) for m in tmp]
        self.flag=True

    def check_flag(self,center=0):
        anum=self.MOLECULE.GetNumAtoms()
        crd=self.MOLECULE.GetConformer().GetPositions()
        dis_array=[]
        for i in range(anum):
            dis_array.append(np.sqrt(np.sum((crd[i]-crd[center])**2)))
        max_dis=np.sort(dis_array)[-1]
        print (max_dis)
        if max_dis>15:
            self.flag=False
        
        #self.Mol=self.Trans2TMMOL()
    def Trans2TrainPoint(self,P):
        anum=self.MOLECULE.GetNumAtoms()
        crd=self.MOLECULE.GetConformer().GetPositions()
        element_list=[]
        for i in self.MOLECULE.GetAtoms():
            element_list.append(i.GetAtomicNum())
        Force=0;Energy=0;Charge=0
        for i in element_list:
            if i==1:
                Energy=Energy+0.496665677286*627.51
            if i==8:
                Energy=Energy+74.9347541416*627.51
            if i==30: 
                Energy=Energy+1779.03302611*627.51
        Energy=Energy+self.ENERGY
        self.FORCE=np.array(self.FORCE)
        self.CHARGE=np.array(self.CHARGE)
        if len(self.FORCE)==anum:
            Force=self.FORCE/627.51
        if self.ENERGY!=0:
            Energy=Energy/627.51
        if len(self.CHARGE)==anum:
            Charge=self.CHARGE
        train_crd=np.zeros((P.MAX_NUM,3),dtype=float)
        train_F=np.zeros((P.MAX_NUM,3),dtype=float)
        train_Q=np.zeros(P.MAX_NUM,dtype=float)
        train_E=0.0
        Trans=np.zeros(P.MAX_NUM,dtype=int)
        Mask=np.zeros(P.MAX_NUM,dtype=float)
        Tnatom=np.zeros(len(P.ATYPE),dtype=int)
        for i in range(anum):
            for j in range(len(P.ATYPE)):
                if (element_list[i]==P.ATYPE[j]):
                    train_crd[P.APT[j][0]+Tnatom[j]]=crd[i]
                    if len(self.FORCE)==anum:
                        train_F[P.APT[j][0]+Tnatom[j]]=Force[i]
                    if Energy!=0:
                        train_E=Energy
                    if len(self.CHARGE)==anum:
                        train_Q[P.APT[j][0]+Tnatom[j]]=Charge[i]
                    Mask[P.APT[j][0]+Tnatom[j]]=1.0
                    Trans[i]=P.APT[j][0]+Tnatom[j]
                    Tnatom[j]=Tnatom[j]+1
        train_point=TRAINPOINT(train_crd,Mask,Trans,anum,train_F,train_Q,train_E)
        return train_point
    def Trans2TMMOL(self):
        anum=self.MOLECULE.GetNumAtoms()
        crd=self.MOLECULE.GetConformer().GetPositions()
        element_list=[]
        for i in self.MOLECULE.GetAtoms():
            element_list.append(i.GetAtomicNum())
        element_list=np.array(element_list)
        Force=0;Energy=0;Charge=0
        for i in element_list:
            if i==1:
                Energy=Energy
            if i==8:
                Energy=Energy
            if i==30: 
                Energy=Energy
        Energy=(Energy+self.ENERGY)
        mol=Mol(element_list,crd)

        if self.ENERGY!=0:
            mol.properties["energy"]=Energy
            mol.CalculateAtomization()
        if len(self.FORCE)==anum:
            mol.properties["force"]=np.array(self.FORCE)
            mol.properties["gradients"]=-np.array(self.FORCE)
        if len(self.CHARGE)==anum:
            mol.properties["charge"]=np.array(self.CHARGE)
        if len(self.DIPOLE)==3:
            mol.properties["dipole"]=np.array(self.DIPOLE)

        return mol

def dump_data(data,filename):
    file=open(filename,'wb')
    pickle.dump(data,file)
    file.close()
    return

def load_data(filename):
    file=open(filename,'rb')
    data=pickle.load(file)
    file.close()
    return data

def GET_NNPARM(CTRLFILE):
    cfile=open(CTRLFILE,'r')
    cline=cfile.readline()
    MODEL_LIST=[]
    MODEL_NUM=-1
    while cline:
        if 'MODEL' in cline:
            MODEL_NUM=MODEL_NUM+1
            MODEL_LIST.append([])
            cline=cfile.readline()
        if 'NN FOR' in cline:
            subnet=SUBNET_CTRL()
            subnet.NAME=cline.split()[-1]
            cline=cfile.readline()
        else :
            print ("ERROR: NEURAL NETWORK SETTING FILE IS WRONG!---- WITHOUT 'NN for' behind the MODEL")
        if 'HIDDEN_LAYER' in cline:
            var=cline.split()
            subnet.HIDDEN_LAYER=[int(m) for  m in var[1:]]
            cline=cfile.readline()
            while 'G' in cline:
                subnet.GPARA.append(cline.split()[:-3]) 
                subnet.SFACTOR.append([float(m) for m in cline.split()[-3:]]) 
                cline=cfile.readline()
            subnet.GNUM=len(subnet.GPARA)
            subnet.STRUC=[subnet.GNUM]+subnet.HIDDEN_LAYER+[1]
            MODEL_LIST[MODEL_NUM].append(subnet)
    return  MODEL_LIST

def PRINT_NNPARM(MODEL,PARM_HANDLE):
    PARM_HANDLE.write('MODEL: \n')
    for i in range(len(MODEL)):
        PARM_HANDLE.write('NN FOR: %s\n'%MODEL[i].NAME )
        PARM_HANDLE.write('HIDDEN_LAYER: %s' %(' '.join('%2d'%m for m in MODEL[i].STRUC)) )
        for j in range(MODEL[i].GNUM):
            #if 'G2' in MODEL[i].GPARA[j][0]:
            #    PARM_HANDLE.write('%s  %.2f  %.2f  %.2f  %.2f  %.2f\n'\
            #    %tuple(MODEL[i].GPARA[j]))
            #elif 'G4' in MODEL[i].GPARA[j][0]:
            #    PARM_HANDLE.write('%s  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f\n'\
            #    %tuple(MODEL[i].GPARA[j]))
            PARM_HANDLE.write(' '.join(m for m in MODEL[i].GPARA[j])+' '+\
            ' '.join('%.2f'%(m) for m in MODEL[i].SFACTOR[j])+'\n')
    return

def Get_global_control(jsonfile):
    with open(jsonfile,'r') as f:
        dict=json.load(f)
        p=PARAMETER()
        p.UPDATE(dict)
        p.SHOW()
    return p

#MODEL_LIST=GET_NNPARM('SOFM-HDNN_NNPARM.in')
#print(MODEL_LIST[0][0].SFACTOR)
#with open('TEST_PARM.in','w') as f:
#    for i in range(len(MODEL_LIST)):
#        PRINT_NNPARM(MODEL_LIST[i],f)
