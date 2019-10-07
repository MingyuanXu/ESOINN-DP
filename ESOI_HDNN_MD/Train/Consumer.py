from ..Comparm import *
import os

def consumer(Queue):
    import time
    from ..Base import Molnew
    import os
    from TensorMol import MSet 
    print ("Consumer start")
    if GPARAMS.Compute_setting.Traininglevel=="DFTB+":    
        os.environ["OMP_NUM_THREADS"]=GPARAMS.Compute_setting.Ncoresperthreads
        para_path=GPARAMS.Software_setting.Dftbparapath
    input_path='./'+GPARAMS.Compute_setting.Traininglevel+'/Consumer/'
    if not os.path.exists(input_path):
        os.system("mkdir -p "+input_path)
    Trainingset=MSet(GPARAMS.Compute_setting.Traininglevel)
    backupname="%s_backup"%GPARAMS.Compute_setting.Traininglevel
    if os.path.exists('./datasets/%s'%GPARAMS.Compute_setting.Traininglevel):
        Trainingset.Load()
        os.system('cp ./datasets/%s.pdb ./datasets/%s.pdb'%(GPARAMS.Compute_setting,backupname))
    Newaddedset=MSet('Newadded')
    Collectset=MSet("Collect"+GPARAMS.Compute_setting.Traininglevel)
    num=0
    while True:
        ERROR_mols=Queue.get()
        if ERROR_mols==None:
            break
        for i in range(len(ERROR_mols)):
            ERROR_mols[i][0].name="Stage_%d_Mol_%d"%(GPARAMS.Train_setting.Trainstage,num)
            if GPARAMS.Train_setting.Ifwithhelp==False:
                if 'energy' not in ERROR_mols[i][0].properties.keys():
                    if GPARAMS.Compute_setting.Traininglevel=="DFTB3": 
                        ERROR_mols[i][0].Write_DFTB_input(para_path,False,input_path)
                        flag=ERROR_mols[i][0].Cal_DFTB(input_path)
                    else:
                        ERROR_mols[i][0].Write_Gaussian_input(GPARAMS.Compute_setting.Gaussiankeywords,input_path,GPARAMS.Compute_setting.Ncoresperthreads,600)
                        flag=ERROR_mols[i][0].Cal_Gaussian(input_path)
                if flag==True:
                    Trainingset.mols.append(ERROR_mols[i][0])
                    Newaddedset.mols.append(ERROR_mols[i][0])
            else:
                if 'energy' in ERROR_mols[i][0].properties.keys():
                    Trainingset.mols.append(ERROR_mols[i][0])
                    Newaddedset.mols.append(ERROR_mols[i][0])
                else:
                    Collectset.mols.append(ERROR_mols[i][0])
            num+=1
            if num>2000:
                num=0
                Trainingset.Save()
                Newaddedset.Save()
                Collectset.Save()
    Trainingset.Save()
    Newaddedset.Save()
    Collectset.Save()
