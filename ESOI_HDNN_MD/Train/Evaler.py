import numpy as np                     
from ESOI_HDNN_MD.Computemethod import Qmmm
from ESOI_HDNN_MD.Comparm import GPARAMS
from ESOI_HDNN_MD.Base.Info import List2str
from ESOI_HDNN_MD import UpdateGPARAMS,LoadModel
from ESOI_HDNN_MD.Train import productor,consumer
from ESOI_HDNN_MD.Train import consumer
from ESOI_HDNN_MD.Computemethod import Cal_NN_EFQ 
import os
#from TensorMol import *
from TensorMol import MSet,JOULEPERHARTREE
import argparse as arg
from multiprocessing import Queue,Process,Manager,Pool
from matplotlib import pyplot as plt 
def evaler():
    path="./results/Stage%d/"%GPARAMS.Train_setting.Trainstage 
    if not os.path.exists(path):
        os.system("mkdir %s"%path)
    rmse=[]
    TMPSet=MSet(GPARAMS.Compute_setting.Traininglevel)
    TMPSet.Load()
    f1=open(path+GPARAMS.Compute_setting.Traininglevel+'.result','w')
    f2=open(path+GPARAMS.Compute_setting.Traininglevel+'_e.csv','w')
    f3=open(path+GPARAMS.Compute_setting.Traininglevel+'_f.csv','w')
    f4=open(path+GPARAMS.Compute_setting.Traininglevel+'_d.csv','w')
    f5=open(path+GPARAMS.Compute_setting.Traininglevel+'_q.csv','w')
    for j in range(len(TMPSet.mols)):
        EGCM=(TMPSet.mols[j].Cal_EGCM()-GPARAMS.Esoinn_setting.scalemin)/(GPARAMS.Esoinn_setting.scalemax-GPARAMS.Esoinn_setting.scalemin)
        EGCM[~ np.isfinite(EGCM)]=0
        TMPSet.mols[j].belongto=GPARAMS.Esoinn_setting.Model.find_closest_cluster(3,EGCM)
    NNpredict,ERRORmols,Avgerr,ERROR_str,method=Cal_NN_EFQ(TMPSet)
    for j in range(len(TMPSet.mols)):
        NNe=NNpredict[j][0]/627.51
        NNf=NNpredict[j][1]/627.51
        NNq=NNpredict[j][3]
        NNd=NNpredict[j][2]
        refe=TMPSet.mols[j].properties["atomization"]
        reff=TMPSet.mols[j].properties["force"]
        refd=TMPSet.mols[j].properties["dipole"]
        try:
            refq=TMPSet.mols[j].properties["resp_charge"]
        except:
            refq=TMPSet.mols[j].properties["charge"]
        rmsde=refe-NNe
        df=np.sqrt(np.sum(np.square(reff-NNf),axis=1))
        maxdf=np.sort(df)[0]
        rmsdf=np.mean(df)
        dd=refd-NNd 
        rmsdd=np.sqrt(np.sum(np.square(refd-NNd)))
        maxdd=np.sort(dd)[0]
        f1.write("%d %s Deviation E:%.3f F Max:%.3f Rmse %.3f D Max: %.3f Rmse %.3f\n"%(j,TMPSet.mols[j].name,rmsde*627.51,maxdf*627.51,rmsdf*627.51,maxdd,rmsdd))
        f2.write("%.3f %.3f\n"%(refe*627.51,NNe*627.51))
        for k in range(len(TMPSet.mols[j].atoms)):
            for l in range(3):
                f3.write("%.3f %.3f\n"%(reff[k][l]*627.51,NNf[k][l]*627.51))
        for k in range(3):
                f4.write("%.3f %.3f\n"%(refd[k],NNd[k]))
    fdata=np.loadtxt("%s"%(path+GPARAMS.Dataset_setting.Inputdatasetlist[i]+'_f.csv'))
    a=fdata[:][0]
    b=fdata[:][1]
    rmse.append(np.sqrt(np.sum((a-b)**2))/len(a))
    edata=np.loadtxt("%s"%(path+GPARAMS.Dataset_setting.Inputdatasetlist[i]+'_e.csv'))
    a=edata[:][0]
    b=edata[:][1]
    rmse.append(np.sqrt(np.sum((a-b)**2))/len(a))
    file=open(path+'rmse.result','w')
    file.write('F rmse: %f\n '%rmse[0])
    file.write('E rmse: %f\n '%rmse[1])
    file.close()
    return rmse