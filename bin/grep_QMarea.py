import numpy as np
from parmed.amber import AmberParm
from parmed.amber import AmberMdcrd
from parmed.amber import Rst7
from ESOI_HDNN_MD.Computemethod import Qmmm
from ESOI_HDNN_MD.Comparm import Parm 
from ESOI_HDNN_MD.Base.Info import List2str
import os
from TensorMol import *
import argparse as arg

if __name__=="__main__":
    parser=arg.ArgumentParser(description='Grep qm area from an Amber MDcrd trajory to make training dataset!')
    parser.add_argument('-i','--input')
    args=parser.parse_args()
    jsonfile=args.input
    Parm.Update(jsonfile)
    os.environ["CUDA_VISIBLE_DEVICES"]=List2str(Parm.Compute_setting.Gpulist,',')
    print (os.environ["CUDA_VISIBLE_DEVICES"])
    TMMSET=MSet(Parm.Dataset_setting.Outputdataset)
    for i in range(len(Parm.System_setting)):
        Strucdict=Parm.System_setting[i].Strucdict
        sys=Qmmm.QMMM_FragSystem(Parm.System_setting[i].Systemparm,\
                                Parm.System_setting[i].Initstruc,\
                                './',Strucdict)
        print (sys.natom)
        natom=sys.natom
        crd=AmberMdcrd(Parm.System_setting[i].Traj,natom,False)
        sys.Create_DisMap()
    
        for i in range(crd.frame):
            sys.step=i+1
            tmpcrd=crd.coordinates[i]
            sys.coords=tmpcrd
            sys.Update_DisMap()
            sys.update_crd()
            tmpmol=sys.Create_QMarea()
            print (tmpmol)
            TMMSET.mols.append(tmpmol)
        TMMSET.Save()

