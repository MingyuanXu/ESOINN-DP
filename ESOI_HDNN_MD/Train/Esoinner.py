import numpy as np
import pickle
import os,sys
from TensorMol import MSet
import math

def esoinner(MSetname=''):
    from ..Comparm import GPARAMS 
    if_continue=True
    if len(GPARAMS.Esoinn_setting.Model.nodes)!=0: 
        cluster_center_before=GPARAMS.Esoinn_setting.Model.cal_cluster_center()
    else:
        cluster_center_before=None 
    if MSetname:
        TotalMSet=MSet(MSetname)
    else:
        TotalMSet=MSet(GPARAMS.Compute_setting.Traininglevel)
    TotalMSet.Load()
    print (len(TotalMSet.mols))
    for i in TotalMSet.mols:
        try:
            i.EGCM
        except:
            i.Cal_EGCM()
    TotalMSet.Save()
    Dataset=np.array([i.EGCM for i in TotalMSet.mols])
    print (Dataset)
    try: 
    #if True:
        if len(GPARAMS.Esoinn_setting.scalemax)==0 and len(GPARAMS.Esoinn_setting.scalemin)==0:
            print("++++++++++++++++++++++++++++++++++++++++++++++++")
            print("initialize the Scalefactor!!!")
            print("++++++++++++++++++++++++++++++++++++++++++++++++")
            GPARAMS.Esoinn_setting.scalemax=np.max(Dataset,0)
            GPARAMS.Esoinn_setting.scalemin=np.min(Dataset,0)
            with open("Sfactor.in",'wb') as f:
                pickle.dump((GPARAMS.Esoinn_setting.scalemax,GPARAMS.Esoinn_setting.scalemin),f)
    except:
        pass

    Dataset=(Dataset-GPARAMS.Esoinn_setting.scalemin)/(GPARAMS.Esoinn_setting.scalemax-GPARAMS.Esoinn_setting.scalemin)
    Dataset[~np.isfinite(Dataset)]=0
    if len(GPARAMS.Esoinn_setting.Model.nodes)!=0:
        Noiseset,a,b,c,d=GPARAMS.Esoinn_setting.Model.predict(Dataset)
    else:
        Noiseset=Dataset 

    GPARAMS.Esoinn_setting.Model.fit(Noiseset,iteration_times=GPARAMS.Train_setting.Esoistep,if_reset=False)
    GPARAMS.Esoinn_setting.Model.Save()
    Noiseset,Noiseindex,nodelabel,cluster_label,signalmask=GPARAMS.Esoinn_setting.Model.predict(Dataset)
    signal_cluster=[[] for i in range(GPARAMS.Esoinn_setting.Model.class_id)]
    
    for i in range(len(Dataset)):
        signal_cluster[cluster_label[i][0]].append(Dataset[i])
    signal_num_list=[len(i) for i in signal_cluster]
    judgenum=math.ceil(sum(signal_num_list)*0.2)
    print ("signal_num_list:",signal_num_list,"judgenum",judgenum)

    #removecluster=[i for i in range(len(signal_num_list)) if not(signal_num_list[i] > judgenum)]
    #print ("removeclusteid:",removecluster)

    #GPARAMS.Esoinn_setting.Model.cut_cluster(removecluster)
    GPARAMS.Esoinn_setting.Model.Save()
    print (GPARAMS.Esoinn_setting.Model.Name,GPARAMS.Esoinn_setting.Model.class_id)  
    print("Class id after Cut action:",GPARAMS.Esoinn_setting.Model.class_id)

    cluster_center_after=GPARAMS.Esoinn_setting.Model.cal_cluster_center()
    if cluster_center_before!=None:# and GPARAMS.Esoinn_setting.NNdict["NN"]!=None:
        print ("Update HDNN")
        updaterule=np.zeros(GPARAMS.Esoinn_setting.Model.class_id)
        for i in range(len(cluster_center_after)):
            vec1=cluster_center_after[i]
            dis=np.sum((np.array(cluster_center_before)-np.array([vec1]*len(cluster_center_before)))**2,1) 
            index=np.argmin(dis)
            print (i,index,"+++++++++++++++++++++++++++")
            updaterule[i]=index 
        """
       # os.system("mkdir -p ./networks/lastsave")
       # os.system("mv ./networks/%s* ./networks/lastsave"%GPARAMS.Esoinn_setting.efdnetname) 
       # os.system("rm `find -name events*`")
       # for i in range(len(cluster_center_after)):
       #     snetname=GPARAMS.Esoinn_setting.efdnetname+"%d_ANI1_Sym_Direct_RawBP_EE_Charge_DipoleEncode_Update_vdw_DSF_elu_Normalize_Dropout_0"%updaterule[i]
       #     tnetname=GPARAMS.Esoinn_setting.efdnetname+"%d_ANI1_Sym_Direct_RawBP_EE_Charge_DipoleEncode_Update_vdw_DSF_elu_Normalize_Dropout_0"%i
       #     os.system("cp ./networks/lastsave/%s.tfn  ./networks/%s.tfn"%(snetname,tnetname))
       #     os.system("cp ./networks/lastsave/%s ./networks/%s -r"%(snetname,tnetname))
       #     for j in ['.index','.meta','.data-00000-of-00001']:
       #         os.system('mv ./networks/%s/%s-chk%s ./networks/%s/%s-chk%s'%(tnetname,snetname,j,tnetname,tnetname,j))
       """
    
