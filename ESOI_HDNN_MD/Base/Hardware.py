import os
import numpy as np

def Find_useable_gpu(gpulist):
    with os.popen("nvidia-smi -q -d Memory |grep -A4 GPU |grep Free",'r')  as f:
        gpumem=[int(i.split()[2]) for i in f.readlines()]
        gpusort=np.argsort(gpumem)
        print (gpusort)
        Flag=True;i=0
        while Flag and i <len(gpusort):
            if gpusort[i] in gpulist:
                Flag=False
                gpustr=str(i)
            i = i+1
        if i>=len(gpusort):
            gpustr=""
    return gpustr
