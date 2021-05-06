import os
import numpy as np

def Find_useable_gpu(gpulist):
    with os.popen("nvidia-smi -q -d Memory |grep -A4 GPU |grep Free",'r')  as f:
        gpumem=[-int(i.split()[2]) for i in f.readlines()]
        print (gpumem)
        gpusort=np.argsort(gpumem)
        print (gpusort)
        Flag=True;i=0
        while Flag and i <len(gpusort):
            if gpusort[i] in gpulist:
                Flag=False
                gpustr=str(gpusort[i])
            i = i+1
        if i>len(gpusort):
            gpustr=""
    return gpustr

def DownloadPDB(namefile,outpath):
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    os.chdir(outpath)
    inputfile = open(namefile,'r')
    for eachline in inputfile:
        pdbname = eachline.lower().strip()[0:4]
        os.system("wget http://ftp.wwpdb.org/pub/pdb/data/structures/all/pdb/pdb" + pdbname + ".ent.gz")
        os.system("gzip -d pdb" + pdbname + '.ent.gz')
        os.system("mv pdb" + pdbname + ".ent " + pdbname.upper() + '.pdb')
    inputfile.close()

