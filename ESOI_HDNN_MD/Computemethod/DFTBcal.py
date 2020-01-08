#!/usr/bin/env python
# coding=utf-8
from ..Base import *

def Cal_DFTB_EFQ(NNSet,parapath,inpath):
    NN_predict=[];ERROR_mols=[];ERROR_strlist=[]
    for i,imol in enumerate(NNSet.mols):
        NNSet.mols[i].Write_DFTB_input(parapath,False,inpath)
        NNSet.mols[i].Cal_DFTB(inpath)
        E_avg=NNSet.mols[i].properties['atomization']*627.51
        F_avg=NNSet.mols[i].properties['force']*627.51
        D_avg=NNSet.mols[i].properties['dipole']
        Q_i=NNSet.mols[i].properties['charge']
        ERROR_mols.append([NNSet.mols[i],999])
        method='DFTB'
        NN_predict.append([E_avg,F_avg,D_avg,Q_i])
        ERROR_strlist.append('')
    return NN_predict,ERROR_mols,0,ERROR_strlist,method

def Cal_Gaussian_EFQ(NNSet,inpath,keywords,nproc=14,mem=600):
    NN_predict=[];ERROR_mols=[];ERROR_strlist=[]
    for i in range(len(NNSet.mols)):
        NNSet.mols[i].Write_Gaussian_input(keywords,inpath,nproc,mem)
        NNSet.mols[i].Cal_Gaussian(inpath)
        NNSet.mols[i].CalculateAtomization(GPARAMS.Compute_setting.Atomizationlevel)
        E_avg=NNSet.mols[i].properties['atomization']*627.51
        F_avg=NNSet.mols[i].properties['force']*627.51
        D_avg=NNSet.mols[i].properties['dipole']
        Q_i=NNSet.mols[i].properties['charge']
        ERROR_mols.append([NNSet.mols[i],999])
        method="Gaussian"
        NN_predict.append([E_avg,F_avg,D_avg,Q_i])
        ERROR_strlist.append('')
    return NN_predict,ERROR_mols,0,ERROR_strlist,method 

