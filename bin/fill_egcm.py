from __future__ import absolute_import
from __future__ import print_function
import sys
sys.path.append('../Esoinn')

from  TensorMol import *
import random
import numpy as np
from ESOI_HDNN_MD.Base import Molnew 
from ESOI_HDNN_MD.Comparm import *
from ESOI_HDNN_MD.Base import Dumpobj
import argparse as arg

if __name__=="__main__":
    parser=arg.ArgumentParser(description='Grep qm area from an Amber MDcrd trajory to make training dataset!')
    parser.add_argument('-i','--input')
    args=parser.parse_args()
    jsonfile=args.input
    Parm.Update(jsonfile)
    OutputMSet=MSet(Parm.Dataset_setting.Outputdataset)
    InputMSetlist=[]
    for i in Parm.Dataset_setting.Inputdatasetlist:
        TMPSet=MSet(i)
        TMPSet.Load()
        OutputMSet.mols+=TMPSet.mols
    OutputMSet.Save()
    d=1
    cycle=math.ceil(len(OutputMSet.mols)/d)
    print (cycle,d)
    EGCMlist=[]
    for i in range(1,cycle+1):
        for j in range(d):
            tmpmol=OutputMSet.mols[(i-1)*d+j]
            try:
                EGCMlist.append(tmpmol.EGCM)
            except:
                tmpmol.Cal_EGCM()
                EGCMlist.append(tmpmol.EGCM)
        Dumpobj(EGCMlist,Parm.Dataset_setting.ESOINNdataset)
    OutputMSet.Save()
