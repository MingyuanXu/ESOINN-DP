from   .Neuralnetwork import *
from   .Comparm import GPARAMS 
import pickle
import random 

def UpdateGPARAMS(jsonfile):
    with open(jsonfile,'r') as f:
        jsondict=json.load(f)
        if 'Compute' in jsondict.keys():
            Loaddict2obj(jsondict['Compute'],GPARAMS.Compute_setting)
            print (GPARAMS.Compute_setting.Traininglevel)
            GPARAMS.Compute_setting.Update() 
        if 'ESOINN' in jsondict.keys():
            Loaddict2obj(jsondict['ESOINN'],GPARAMS.Esoinn_setting)
            GPARAMS.Esoinn_setting.Update()
            nnlist=None;respnet=None
            if GPARAMS.Esoinn_setting.Modelfile!="":
                GPARAMS.Esoinn_setting.Model=Esoinn(GPARAMS.Esoinn_setting.Modelfile,\
                                                    dim=GPARAMS.Esoinn_setting.Maxnum,\
                                                    iteration_threshold=GPARAMS.Esoinn_setting.Traininterval)
                if os.path.exists(GPARAMS.Esoinn_setting.Modelfile+".ESOINN"):
                    GPARAMS.Esoinn_setting.Model.Load()
                    print (GPARAMS.Esoinn_setting.Model.nodes)
            if GPARAMS.Esoinn_setting.Scalefactorfile!="":
                if os.path.exists(GPARAMS.Esoinn_setting.Scalefactorfile):
                    with open(GPARAMS.Esoinn_setting.Scalefactorfile,'rb') as f:
                        GPARAMS.Esoinn_setting.scalemax,GPARAMS.Esoinn_setting.scalemin=pickle.load(f)
                else:
                    GPARAMS.Esoinn_setting.scalemax=None
                    GPARAMS.Esoinn_setting.scalemin=None
            if GPARAMS.Esoinn_setting.Loadefdnet==True and GPARAMS.Esoinn_setting.Model!=None:
                nnlist=Get_neuralnetwork_instance(GPARAMS.Esoinn_setting.Model.class_id)
            if GPARAMS.Esoinn_setting.Loadrespnet==True and GPARAMS.Esoinn_setting.respnetname!="":
                respnet=Get_resp_instance(GPARAMS.Esoinn_setting.respnetname)
            GPARAMS.Esoinn_setting.NNdict={"NN":nnlist,"RESP":respnet}

        if 'HDNN' in jsondict.keys():
            Loaddict2obj(jsondict['HDNN'],GPARAMS.Neuralnetwork_setting)
            GPARAMS.Neuralnetwork_setting.Update()
        if 'System' in jsondict.keys():
            for i in range(len(jsondict["System"])):
                Tmpsys_setting=System_setting()
                Loaddict2obj(jsondict['System'][i],Tmpsys_setting)
                GPARAMS.System_setting.append(Tmpsys_setting)
        if 'MD' in jsondict.keys():
            for i in range(len(jsondict["MD"])):
                Tmpmd_setting=MD_setting()
                Loaddict2obj(jsondict["MD"][i],Tmpmd_setting)
                GPARAMS.MD_setting.append(Tmpmd_setting)
        if "Dataset" in jsondict.keys():
            Loaddict2obj(jsondict['Dataset'],GPARAMS.Dataset_setting)
        if "Train" in jsondict.keys():
            Loaddict2obj(jsondict["Train"],GPARAMS.Train_setting)
            if GPARAMS.Train_setting.Trainstage!=0:
                GPARAMS.MD_setting.Stageindex=GPARAMS.Train_setting.Trainstage
                GPARAMS.MD_setting.Mdstage=GPARAMS.Train_setting.Stagenum
                
    return

def Loaddict2obj(dict,obj):
    objdict=obj.__dict__
    for i in dict.keys():
        if i not in objdict.keys():
            print ("%s not is not a standard setting option!"%i)
        objdict[i]=dict[i]
    obj.__dict__=objdict
    #print (obj.Modelfile)

def Get_neuralnetwork_instance(Class_num):
    subnet_list=[]
    for i in range(Class_num):
        SUBNET=BP_HDNN(None,'Cluster%i_ANI1_Sym_Direct_RawBP_EE_Charge_DipoleEncode_Update_vdw_DSF_elu_Normalize_Dropout_0'%i,False)
        subnet_list.append(SUBNET)
        #SUBNET.SaveAndClose()
    return subnet_list

def Get_resp_instance(Name):
    SUBNET=BP_HDNN_charge(None,Name,False)
    #SUBNET.SaveAndClose()
    return SUBNET

