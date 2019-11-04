#!/usr/bin/env python
# coding=utf-8
from .DFTBcal import *
from ..Neuralnetwork import *
from ..Base import *
from ..Comparm import *
import numpy as np

def Cal_NN_EFQ(NNSet,inpath='./'):
    ERROR_mols=[]
    MSet_list=[MSet('ID%d'%i) for i in range(len(GPARAMS.Esoinn_setting.NNdict['NN']))]
    Mol_label=[[] for i in NNSet.mols]
    if GPARAMS.Esoinn_setting.NNdict["RESP"]!=None:
        N_Times=math.ceil(len(NNSet.mols)/GPARAMS.Neuralnetwork_setting.Batchsize)
        RESPCHARGE=[]
        for i in range(N_Times):
            TMMSET=MSet('tmp')
            TMMSET.mols=NNSet.mols[i*GPARAMS.Neuralnetwork_setting.Batchsize:(i+1)*GPARAMS.Neuralnetwork_setting.Batchsize]
            #try:
            atom_charge=\
                    Eval_charge(TMMSET,GPARAMS.Esoinn_setting.NNdict["RESP"])
            #except:
            #    atom_charge=[]
            RESPCHARGE+=list(atom_charge)
    for i in range(len(NNSet.mols)):
        for j in NNSet.mols[i].belongto:
            MSet_list[j].mols.append(NNSet.mols[i])
            Mol_label[i].append([j,len(MSet_list[j].mols)-1])
    E=[];F=[];Dipole=[];Charge=[]
    for i in range(len(GPARAMS.Esoinn_setting.NNdict["NN"])):
        if len(MSet_list[i].mols)>0:
            N_Times=math.ceil(len(MSet_list[i].mols)/GPARAMS.Neuralnetwork_setting.Batchsize)
            E_tmp=[];F_tmp=[];Dipole_tmp=[];Charge_tmp=[]
            for j in range(N_Times):
                TMMSET=MSet('tmp')
                TMMSET.mols=MSet_list[i].mols[j*GPARAMS.Neuralnetwork_setting.Batchsize:(j+1)*GPARAMS.Neuralnetwork_setting.Batchsize]
                #print ("NN Calculation at here!")
                Etotal,Ebp,Ebp_atom,Ecc,Evdw,mol_dipole,atom_charge,gradient=\
                    EvalSet(TMMSET,GPARAMS.Esoinn_setting.NNdict["NN"][i])
                #print ("NN Calculation at over!")
                E_tmp+=list(Etotal);F_tmp+=list(gradient);Dipole_tmp+=list(mol_dipole);Charge_tmp+=list(atom_charge)
            E.append(E_tmp)
            F.append(F_tmp)
            Dipole.append(Dipole_tmp)
            Charge.append(Charge_tmp)
        else:
            E.append([])
            F.append([])
            Dipole.append([])
            Charge.append([])
    MAX_ERR=[]
    NN_predict=[]
    ERROR_str=''
    for i,imol in enumerate(NNSet.mols):
        E_i=[];F_i=[];D_i=[];Q_i=[]
        for j in Mol_label[i]:
            E_i.append(E[j[0]][j[1]])
            F_i.append(F[j[0]][j[1]][0:len(imol.coords)])
            D_i.append(Dipole[j[0]][j[1]])
            Q_i.append(Charge[j[0]][j[1]][0:len(imol.coords)])
        E_i=np.array(E_i)*627.51
        F_i=np.array(F_i)*627.51/JOULEPERHARTREE
        D_i=np.array(D_i)
        Q_i=np.array(Q_i)
        NN_num=len(imol.belongto)
        if NN_num <=3:
            N_num=min(2,NN_num)
        else:
            N_num=math.ceil((NN_num+1)/2)
        E_avg=np.mean(E_i)
        F_avg=np.mean(F_i,axis=0)
        tmp_list=np.argsort(np.max(np.reshape(np.square(F_i-F_avg),(len(imol.belongto),-1)),1))[:N_num]
        F_New=[F_i[m] for m in tmp_list]
        F_avg=np.mean(F_New,axis=0)
        E_New=[E_i[m] for m in tmp_list]
        D_New=[D_i[m] for m in tmp_list]
        Q_New=[Q_i[m] for m in tmp_list]
        E_avg=np.mean(E_New)
        D_avg=np.mean(D_New,axis=0)
        Q_avg=np.mean(Q_New,axis=0)
        MSE_F=np.square(F_New-F_avg).mean(axis=0)
        MAX_MSE_F=-np.sort(-np.reshape(MSE_F,-1))[0]
        MAX_ERR.append(MAX_MSE_F)
        method='NN'
        if MAX_MSE_F > GPARAMS.Neuralnetwork_setting.Maxerr :
            ERROR_str+='%dth mol in NNSet is not believable, MAX_MSE_F: %f\n '%(i,MAX_MSE_F)
            print(ERROR_str)
            ERROR_mols.append([NNSet.mols[i],MAX_MSE_F])
#         
#        if MAX_MSE_F>=50 or MAX_MSE_F-tmperr>30:
#            ERROR_str+='%dth mol will be calculated with DFTB!'
#            NNSet.mols[i].Write_DFTB_input(parapath,False,inpath)
#            NNSet.mols[i].Cal_DFTB(inpath)
#            E_avg=NNSet.mols[i].properties['energy']*627.51
#            F_avg=NNSet.mols[i].properties['force']*627.51
#            try:
#                D_avg=NNSet.mols[i].properties['dipole']
#            except:
#                D_avg=np.zeros(3)
#            Q_i=NNSet.mols[i].properties['charge']
#            NNSet.mols[i].properties={}
#            ERROR_mols.append(NNSet.mols[i])
#            method='DFTB'
#
        NN_predict.append([E_avg,F_avg,D_avg,Q_avg])
    AVG_ERR=np.mean(np.array(MAX_ERR))
    if GPARAMS.Esoinn_setting.NNdict["RESP"]!=None:
        for i in range(len(NNSet.mols)):
            NN_predict[i][3]=RESPCHARGE[i]
    return NN_predict,ERROR_mols,AVG_ERR,ERROR_str,method

