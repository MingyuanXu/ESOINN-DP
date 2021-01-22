from   .Neuralnetwork import *
from   .Comparm import GPARAMS 
import pickle
import random 

def UpdateGPARAMS(jsonfile,Onlymodel=False):
    with open(jsonfile,'r') as f:
        jsondict=json.load(f)
        if 'ESOINN' in jsondict.keys():
            Loaddict2obj(jsondict['ESOINN'],GPARAMS.Esoinn_setting)
            GPARAMS.Esoinn_setting.Update()
        if 'Compute' in jsondict.keys():
            Loaddict2obj(jsondict['Compute'],GPARAMS.Compute_setting)
            print (GPARAMS.Compute_setting.Gaussiankeywords)
            GPARAMS.Compute_setting.Update() 
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
            GPARAMS.Train_setting.Update()
            print ("**************************************")
            print (GPARAMS.Train_setting.sigma)
            print ("**************************************")
            if GPARAMS.Train_setting.Trainstage!=0:
                for i in range(len(GPARAMS.MD_setting)):
                    GPARAMS.MD_setting[i].Stageindex=GPARAMS.Train_setting.Trainstage
                    GPARAMS.MD_setting[i].Mdstage=GPARAMS.Train_setting.Stagenum
    return

def LoadModel(ifhdnn=True):
    nnlist=None;chargenet=None
    if GPARAMS.Esoinn_setting.Modelfile!="":
        GPARAMS.Esoinn_setting.Model=Esoinn(GPARAMS.Esoinn_setting.Modelfile,\
                                            dim=GPARAMS.Esoinn_setting.Maxnum,\
                                            iteration_threshold=GPARAMS.Esoinn_setting.Traininterval)
        if os.path.exists(GPARAMS.Esoinn_setting.Modelfile+".ESOINN"):
            GPARAMS.Esoinn_setting.Model.Load()
        print (GPARAMS.Esoinn_setting.Model.class_id)
    if GPARAMS.Esoinn_setting.Scalefactorfile!="":
        if os.path.exists(GPARAMS.Esoinn_setting.Scalefactorfile):
            with open(GPARAMS.Esoinn_setting.Scalefactorfile,'rb') as f:
                GPARAMS.Esoinn_setting.scalemax,GPARAMS.Esoinn_setting.scalemin=pickle.load(f)
                #print (GPARAMS.Esoinn_setting.scalemax,GPARAMS.Esoinn_setting.scalemin)
        else:
            GPARAMS.Esoinn_setting.scalemax=None
            GPARAMS.Esoinn_setting.scalemin=None
    if ifhdnn==True:
        if GPARAMS.Esoinn_setting.Loadefdnet==True and GPARAMS.Esoinn_setting.Model!=None:
            print ("===========start load net===============")
            nnlist=Get_neuralnetwork_instance(max(GPARAMS.Esoinn_setting.Model.class_id,GPARAMS.Train_setting.Modelnumperpoint))
        if GPARAMS.Esoinn_setting.Loadchargenet==True and GPARAMS.Esoinn_setting.chargenetname!="":
            chargenet=Get_charge_instance(GPARAMS.Esoinn_setting.chargenetname)
        GPARAMS.Esoinn_setting.NNdict={"NN":nnlist,"Charge":chargenet}
    else:
        GPARAMS.Esoinn_setting.NNdict={"NN":nnlist,"Charge":nnlist}

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
        SUBNET=BP_HDNN(None,GPARAMS.Esoinn_setting.efdnetname+'%d_ANI1_Sym_Direct_RawBP_EE_Charge_DipoleEncode_Update_vdw_DSF_elu_Normalize_Dropout_0'%i,False)
        subnet_list.append(SUBNET)
    print ("Loaded %d subnet in Neural Layer!"%(len(subnet_list)))
    return subnet_list

def Get_charge_instance(Name):
    SUBNET=BP_HDNN_charge(None,Name,False)
    #SUBNET.SaveAndClose()
    return SUBNET

def Added_MSet(filename):
    TotalMSet=MSet(GPARAMS.Compute_setting.Traininglevel)
    if os.path.exists('./datasets/'+GPARAMS.Compute_setting.Traininglevel+'.pdb'):
        TotalMSet.Load()
    NewaddedSet=MSet(filename)
    NewaddedSet.Load()
    if GPARAMS.Esoinn_setting.Ifresp==True:
        RespMSet=MSet('HF_resp')
        if os.path.exists('./datasets/HF_resp.pdb'):
            RespMSet.Load()
        RespMSet.mols+=NewaddedSet.mols 
        RespMSet.Save()
    TotalMSet.mols+=NewaddedSet.mols 
    TotalMSet.Save()
#    print (RespMSet.mols[-1],RespMSet.mols[-1].properties)
#    print (TotalMSet.mols[-1],TotalMSet.mols[-1].properties)
    return

