from ..Comparm import *
import os

def productor(ID='',GPARAMS_index=0,Queue=None):
    from ..Computemethod import QMMM_FragSystem
    from ..MD import Simulation
    os.environ["CUDA_VISIBLE_DEVICES"]=ID

    if GPARAMS.Compute_setting.Theroylevel=="DFTB":
        os.environ["OMP_NUM_THREADS"]=GPARAMS.Compute_setting.Ncoresperthreads

    if GPARAMS.Compute_setting.Computelevel[GPARAMS_index]=="QM/MM":
        if GPARAMS.System_setting[GPARAMS_index].Forcefield=="Amber":
            prmfile=GPARAMS.System_setting[GPARAMS_index].Systemparm
            MDpath='./'+GPARAMS.MD_setting[GPARAMS_index].Name+'/'
            os.system("cp "+prmfile+' '+MDpath+'/'+prmfile)
            if GPARAMS.MD_setting[GPARAMS_index].Stageindex!=0:
                restartstruc=GPARAMS.MD_setting[GPARAMS_index].Name+\
                        '_%d.rst7'%(GPARAMS.MD_setting[GPARAMS_index].Stageindex-1)
                initstruc=GPARAMS.MD_setting[GPARAMS_index].Name+\
                        '_%d.inpcrd'%(GPARAMS.MD_setting[GPARAMS_index].Stageindex)
                os.system('cp '+MDpath+restartstruc+' '+MDpath+initstruc)
            else:
                initstruc=GPARAMS.System_setting[GPARAMS_index].Initstruc
                os.system("cp "+initstruc+' '+MDpath+initstruc)
            qmsys=QMMM_FragSystem(MDpath+prmfile,MDpath+initstruc,\
                                Strucdict=GPARAMS.System_setting[GPARAMS_index].Strucdict,\
                                Path=GPARAMS.MD_setting[GPARAMS_index].Name)
    if GPARAMS.MD_setting[GPARAMS_index].MDmethod=="Normal MD":
        MD_simulation=Simulation(sys=qmsys,\
                                 MD_setting=GPARAMS.MD_setting[GPARAMS_index])
        MDdeviation=MD_simulation.MD(Queue)

