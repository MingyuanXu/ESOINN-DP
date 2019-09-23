from ..Comparm import *
import os

def consumer(Queue):
    import time
    from ..Base import Molnew
    import os
    from TensorMol import MSet 
    
    if GPARAMS.Compute_setting.Traininglevel=="DFTB+":    
        os.environ["OMP_NUM_THREADS"]=GPARAMS.Compute_setting.Ncoresperthreads
        para_path=GPARAMS.Software_setting.Dftbparapath
    input_path='./'+GPARAMS.Compute_setting.Traininglevel+'/Consumer'
    if not os.path.exists(input_path):
        os.system("mkdir -p "+input_path)
    Trainingset=MSet(GPARAMS.Compute_setting.Traininglevel)
    backupname="%s_backup"%GPARAMS.Compute_setting.Traininglevel
    if os.path.exists('./datasets/%s'%GPARAMS.Compute_setting.Traininglevel):
        Trainingset.Load()
        os.system('cp ./datasets/%s.pdb ./datasets/%s.pdb'%(GPARAMS.Compute_setting,backupname))
    Newaddedset=MSet('Newadded')
    num=0
    while True:
        ERROR_mols=Queue.get()
        if ERROR_mols==None:
            break
        for i in range(len(ERROR_mols)):
            if 'energy' not in ERROR_mols[i].properties.keys():
                if GPARAMS.Compute_setting.Traininglevel=="DFTB+": 
                    ERROR_mols[i].Write_DFTB_input(para_path,False,input_path)
                    ERROR_mols[i].Cal_DFTB(input_path)
                elif GPARAMS.Compute_setting.Traininglevel=="":
                    pass
            Trainingset.mols.append(ERROR_mols[i])
            Newaddedset.mols.append(ERROR_mols[i])
            num+=1
            if num>2000:
                num=0
                Trainingset.Save()
                Newaddedset.Save()
    Trainingset.Save()
    Newaddedset.Save()

