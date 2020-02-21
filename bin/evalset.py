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
parser=arg.ArgumentParser(description='Grep qm area from an Amber MDcrd trajory to make training dataset!')
parser.add_argument('-i','--input')

args=parser.parse_args()
jsonfile=args.input
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if __name__=="__main__":
    UpdateGPARAMS(jsonfile)
    LoadModel()
    if not os.path.exists("./results") :
        os.system("mkdir ./results")
    for i in range(len(GPARAMS.Dataset_setting.Inputdatasetlist)):
        TMPSet=MSet(GPARAMS.Dataset_setting.Inputdatasetlist[i])
        TMPSet.Load()
        f1=open('./results/'+GPARAMS.Dataset_setting.Inputdatasetlist[i]+'.result','w')
        f2=open('./results/'+GPARAMS.Dataset_setting.Inputdatasetlist[i]+'_e.csv','w')
        f3=open('./results/'+GPARAMS.Dataset_setting.Inputdatasetlist[i]+'_f.csv','w')
        f4=open('./results/'+GPARAMS.Dataset_setting.Inputdatasetlist[i]+'_d.csv','w')
        f5=open('./results/'+GPARAMS.Dataset_setting.Inputdatasetlist[i]+'_q.csv','w')
        for j in range(len(TMPSet.mols)):
        #for j in range(10):
            EGCM=(TMPSet.mols[j].Cal_EGCM()-GPARAMS.Esoinn_setting.scalemin)/(GPARAMS.Esoinn_setting.scalemax-GPARAMS.Esoinn_setting.scalemin)
            EGCM[~ np.isfinite(EGCM)]=0
            TMPSet.mols[j].belongto=GPARAMS.Esoinn_setting.Model.find_closest_cluster(3,EGCM)
            print (j,TMPSet.mols[j].belongto)
        NNpredict,ERRORmols,Avgerr,ERROR_str,method=Cal_NN_EFQ(TMPSet)
        for j in range(len(TMPSet.mols)):
        #for j in range(10):
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
            f1.write("%d %s Deviation E:%.3f F Max:%.3f Rmse %.3f D Max: %.3f Rmse %.3f\n"%(j,TMPSet.mols[j].name,rmsde,maxdf,rmsdf,maxdd,rmsdd))
            f2.write("%.3f %.3f\n"%(refe*627.51,NNe*627.51))
            for k in range(len(TMPSet.mols[j].atoms)):
                for l in range(3):
                    f3.write("%.3f %.3f\n"%(reff[k][l]*627.51,NNf[k][l]*627.51))
            #    f5.write("%.3f %.3f\n"%(refq[k],NNq[k]))
            for k in range(3):
                    f4.write("%.3f %.3f\n"%(refd[k],NNd[k]))
