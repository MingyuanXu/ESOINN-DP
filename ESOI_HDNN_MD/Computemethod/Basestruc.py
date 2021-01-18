#!/usr/bin/env python
# coding=utf-8

import time
import numpy as np
import copy
from ..Comparm import *
#from  TensorMol import *
from parmed.amber import AmberParm
from parmed.amber import AmberMdcrd
from parmed.amber import Rst7
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
print (Table_Element)
class atom:
    def __init__(self,eleindex,aname,aindex,rname,rindex):
        self.eleindex=eleindex
        self.aname=aname
        self.aindex=aindex
        self.rname=rname
        self.rindex=rindex
        self.atomnum=eleindex
        self.element=Element_Table[eleindex]
        self.realpt=aindex 
        self.bpt=-1
    def get_crd(self,crd):
        self.crd=crd   
    def Show_inpdb(self):
        str='ATOM  %5d %4s %3s %5d    %8.3f%8.3f%8.3f%6.2f%6.2f %-2s\n'%(self.aindex+1,self.aname,self.rname,self.rindex+1,self.crd[0],self.crd[1],self.crd[2],1.00,0.00,self.element)
        return str
    def Show_inpdb_with_option(self,rname,rindex):
        str='ATOM  %5d %4s %3s %5d    %8.3f%8.3f%8.3f%6.2f%6.2f %-2s\n'%(self.aindex+1,self.aname,rname,rindex+1,self.crd[0],self.crd[1],self.crd[2],1.00,0.00,self.element)
        return str 

class ghost_atom:
    def __init__(self,eleindex,aname,aindex,rname,rindex,realpt,bpt,ctype):
        atom.__init__(self,eleindex,aname,aindex,rname,rindex)
        self.realpt=realpt
        self.bpt=bpt
        self.ctype=ctype
        if self.ctype=='CC':
            self.bondlength=1.09000
        elif self.ctype=='CN':
            self.bondlength=1.09000
        elif self.ctype=='NC':
            self.bondlength=1.00000
        elif self.ctype=='CO':
            self.bondlength=1.09000
        elif self.ctype=='OC':
            self.bondlength=0.99619
        elif self.ctype=='SC':
            self.bondlength=1.31000
        elif self.ctype=='CS':
            self.bondlength=1.09000
    def get_crd(self,crdbpt,crdrpt):
        dis=np.sqrt(np.sum(np.square(crdrpt-crdbpt)))
        self.crd=self.bondlength*(crdrpt-crdbpt)/dis+crdbpt
    def Show_inpdb(self):
        str='ATOM  %5d %4s %3s %5d    %8.3f%8.3f%8.3f%6.2f%6.2f %-2s\n'%(self.paindex+1,self.paname,self.prname,self.prindex+1,self.crd[0],self.crd[1],self.crd[2],1.00,1.00,self.element)
        return str
    def Show_inpdb_with_option(self,rname,rindex):
        str='ATOM  %5d %4s %3s %5d    %8.3f%8.3f%8.3f%6.2f%6.2f %-2s\n'%(self.aindex+1,self.aname,rname,rindex+1,self.crd[0],self.crd[1],self.crd[2],1.00,0.00,self.element)
        return str 
         
class frag:
    def __init__(self,atomlist,cbnd=[],gtype=[],fragtype='',fragindex=0,fragname='',fragspin=1):
        self.atomlist=atomlist
        self.indexlist=np.array([m.aindex for m in atomlist])
        self.glist=[]
        self.gtype=gtype
        self.fragtype=fragtype
        self.fragindex=fragindex
        self.fragname=fragname
        self.fragspin=fragspin
        
        if len(cbnd)>0:
            self.cbnd=cbnd
            self.cbpti=[cbnd[m][0] for m in range(len(cbnd))]
            self.cbpto=[cbnd[m][1] for m in range(len(cbnd))]
        else:
            self.cbnd=[]
            self.cbpti=[]
            self.cbpto=[]
        tmp=0
        
        for i in range(len(self.cbpti)):
            for j in range(len(atomlist)):
                if atomlist[j].aindex==self.cbpti[i]:
                    if atomlist[j].aname=='CA':
                        tmp=tmp+1
                        gatom=ghost_atom(1,'H%d'%tmp,atomlist[j].aindex-tmp,atomlist[j].rname,atomlist[j].rindex,realpt=self.cbpto[i],bpt=atomlist[j].aindex,ctype=gtype[i])
                        self.glist.append(gatom)
                    if atomlist[j].aname=='C':     
                        gatom=ghost_atom(1,'H1',atomlist[j].aindex-1,atomlist[j].rname,atomlist[j].rindex,realpt=self.cbpto[i],bpt=atomlist[j].aindex,ctype=gtype[i])
                        self.glist.append(gatom)
                    if atomlist[j].aname=='N':
                        gatom=ghost_atom(1,'H2',atomlist[j].aindex+2,atomlist[j].rname,atomlist[j].rindex,realpt=self.cbpto[i],bpt=atomlist[j].aindex,ctype=gtype[i])     
                        self.glist.append(gatom)
    
    def get_crd(self,crd):
        for i in range(len(self.atomlist)):
            self.atomlist[i].get_crd(crd[i])
    
    def Show_inpdb(self):
        str=''
        if self.fragname=='':
            for i in range(len(self.atomlist)):
                str+=self.atomlist[i].Show_inpdb()
        else:
            for i in range(len(self.atomlist)):
                str+=self.atomlist[i].Show_inpdb_with_option(self.fragname,self.fragindex)
        return str
    
    def get_Mol2para(self,charge,path='./'):
        pdbfile=open(path+self.fragname+'.pdb','w')
        str=''
        for i in range(len(self.atomlist)):
            str+=self.atomlist[i].Show_inpdb_with_option(self.fragname,self.fragindex)
        pdbfile.write(str)
        pdbfile.close()
        antechamber='antechamber -i %s -fi pdb -o %s -fo mol2 -c bcc -s 2 -nc %d'%(path+self.fragname+'.pdb',path+self.fragname+'.mol2',charge)
        os.system(antechamber)
        prmchk='parmchk2 -i %s -o %s -f mol2'%(path+self.fragname+'.mol2',path+self.fragname+'.frcmod')
        os.system(prmchk)

