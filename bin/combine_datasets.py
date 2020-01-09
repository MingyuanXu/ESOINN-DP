from __future__ import absolute_import
from __future__ import print_function
import sys
sys.path.append('../Esoinn')

from  TensorMol import *
import random
import numpy as np
from ESOI_HDNN_MD.Base import Molnew 
from ESOI_HDNN_MD.Comparm import *
import argparse as arg

if __name__=="__main__":
    parser=arg.ArgumentParser(description='combine dataset')
    parser.add_argument('-i','--input')
    args=parser.parse_args()
    UpdateGPARAMS(args.input) 
    OutputMSet=MSet(GPARAMS.Dataset_setting.Outputdataset)
    InputMSetlist=[]
    for i in GPARAMS.Dataset_setting.Inputdatasetlist:
        TMPSet=MSet(i)
        TMPSet.Load()
        OutputMSet.mols+=TMPSet.mols
    print (len(OutputMSet.mols))
    OutputMSet.Save()
    
