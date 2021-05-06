import numpy as np                     
from ..Comparm import GPARAMS
from ..Base.Info import List2str
from ..LoadGPARAMS import UpdateGPARAMS,LoadModel
from ..Computemethod import *
import os
import random
#from TensorMol import *
from TensorMol import MSet,JOULEPERHARTREE
import argparse as arg
from multiprocessing import Queue,Process,Manager,Pool
from matplotlib import pyplot as plt 
def evaler(MSetname):
    print ('************************************************')
    print (GPARAMS.Train_setting.sigma)
    print ('************************************************')
    os.environ["CUDA_VISIBLE_DEVICES"]=str(GPARAMS.Compute_setting.Gpulist[0])
    path="./results/Stage%d/%s/"%(GPARAMS.Train_setting.Trainstage,MSetname)  
    if not os.path.exists(path):
        os.system("mkdir -p %s"%path)
    rmse=[]
    #TMPSet=MSet(GPARAMS.Compute_setting.Traininglevel)
    TMPSet=MSet(MSetname)
    TMPSet.Load()
    #TMPSet.mols=random.sample(TMPSet.mols,200)
    #TMPSet.mols=random.sample(TMPSet.mols,200)
    f1=open(path+GPARAMS.Compute_setting.Traininglevel+'.result','w')
    f2=open(path+GPARAMS.Compute_setting.Traininglevel+'_e.csv','w')
    f3=open(path+GPARAMS.Compute_setting.Traininglevel+'_f.csv','w')
    f4=open(path+GPARAMS.Compute_setting.Traininglevel+'_d.csv','w')
    f5=open(path+GPARAMS.Compute_setting.Traininglevel+'_q.csv','w')
    for j in range(len(TMPSet.mols)):
        EGCM=(TMPSet.mols[j].Cal_EGCM()-GPARAMS.Esoinn_setting.scalemin)/(GPARAMS.Esoinn_setting.scalemax-GPARAMS.Esoinn_setting.scalemin)
        EGCM[~ np.isfinite(EGCM)]=0
        TMPSet.mols[j].belongto=GPARAMS.Esoinn_setting.Model.find_closest_cluster(3,EGCM)
        TMPSet.mols[j].properties['clabel']=int(TMPSet.mols[j].totalcharge)
    NNpredict,ERRORmols,Avgerr,ERROR_str,method=Cal_NN_EFQ(TMPSet)
    print (NNpredict)
    for j in range(len(TMPSet.mols)):
        NNe=NNpredict[j][0]/627.51
        NNf=NNpredict[j][1]/627.51
        NNq=NNpredict[j][3]
        NNd=NNpredict[j][2]
        refe=TMPSet.mols[j].properties["atomization"]
        reff=TMPSet.mols[j].properties["force"]
        refd=TMPSet.mols[j].properties["dipole"]
        if GPARAMS.Esoinn_setting.Ifresp==True:
            try:
                print ("RESP charge")
                refq=TMPSet.mols[j].properties["resp_charge"]
            except:
                print ("Other charge")
                refq=TMPSet.mols[j].properties["charge"]
        elif GPARAMS.Esoinn_setting.Ifadch==True:
            try:
                print ("RESP charge")
                refq=TMPSet.mols[j].properties["adch_charge"]
            except:
                print ("Other charge")
                refq=TMPSet.mols[j].properties["charge"]
                
        rmsde=refe-NNe
        print ("HHHHHHHHHJJJJJJJJJJJJJKKKKKKKKKKKK")
        print (NNf,reff)
        print ("HHHHHHHHHJJJJJJJJJJJJJKKKKKKKKKKKK")
        df=np.reshape(np.square(reff-NNf),-1)
        maxdf=np.max(np.sqrt(df))
        rmsdf=np.sqrt(np.sum(df)/len(df))
        dd=refd-NNd 
        rmsdd=np.sqrt(np.sum(np.square(refd-NNd)))
        maxdd=np.sort(dd)[0]
        f1.write("%d %s Deviation E:%.3f F Max:%.3f Rmse %.3f D Max: %.3f Rmse %.3f\n"%(j,TMPSet.mols[j].name,rmsde*627.51,maxdf*627.51,rmsdf*627.51,maxdd,rmsdd))
        f2.write("%.3f %.3f\n"%(refe*627.51,NNe*627.51))
        for k in range(len(TMPSet.mols[j].atoms)):
            for l in range(3):
                f3.write("%.3f %.3f\n"%(reff[k][l]*627.51,NNf[k][l]*627.51))
            if GPARAMS.Esoinn_setting.Ifresp==True or GPARAMS.Esoinn_setting.Ifadch==True:
                f5.write("%.3f %.3f\n"%(refq[k],NNq[k]))
        for k in range(3):
                f4.write("%.3f %.3f\n"%(refd[k],NNd[k]))
        f1.flush()
        f2.flush()
        f3.flush()
        f4.flush()
        if GPARAMS.Esoinn_setting.Ifresp==True or GPARAMS.Esoinn_setting.Ifadch==True:
            f5.flush()
    
    fdata=np.loadtxt("%s"%(path+GPARAMS.Compute_setting.Traininglevel+'_f.csv'))
    a=fdata[:,0]
    b=fdata[:,1]
    rmse.append(np.sqrt(np.sum((a-b)**2)/len(a)))
    edata=np.loadtxt("%s"%(path+GPARAMS.Compute_setting.Traininglevel+'_e.csv'))
    a=edata[:,0]
    b=edata[:,1]
    rmse.append(np.sqrt(np.sum((a-b)**2)/len(a)))
    if GPARAMS.Esoinn_setting.Ifresp==True or GPARAMS.Esoinn_setting.Ifadch==True:
        qdata=np.loadtxt("%s"%(path+GPARAMS.Compute_setting.Traininglevel+'_q.csv'))
        a=qdata[:,0]
        b=qdata[:,1]
        rmse.append(np.sqrt(np.sum((a-b)**2)/len(a)))

    file=open(path+'rmse.result','w')
    file.write('F rmse: %f\n '%rmse[0])
    file.write('E rmse: %f\n '%rmse[1])
    if GPARAMS.Esoinn_setting.Ifresp==True or GPARAMS.Esoinn_setting.Ifadch==True:
        file.write('Q rmse: %f\n '%rmse[2])
    file.close()
    return rmse
