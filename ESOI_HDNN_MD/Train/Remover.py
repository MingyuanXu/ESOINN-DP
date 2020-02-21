from ..Comparm import *
import os
def remover(GPARAMS_index=0,Queue=None,GPUQueue=None):
    from ..Base import Find_useable_gpu
    from ..Computemethod import Cal_NN_EFQ 
    print (GPARAMS.Compute_setting.Traininglevel)
    print (GPARAMS.Compute_setting.Theroylevel)
#    os.environ["CUDA_VISIBLE_DEVICES"]=Find_useable_gpu(GPARAMS.Compute_setting.Gpulist)
    GPUid=GPUQueue.get()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(GPUid)
    print (os.environ["CUDA_VISIBLE_DEVICES"])
    tmpset=MSet(GPARAMS.Dataset_setting.Inputdatasetlist[GPARAMS_index])
    tmpset.Load()
    unknownlist=[]
    interval=5*GPARAMS.Neuralnetwork_setting.Batchsize
    stepnum=math.ceil(len(tmpset.mols)/interval)
    Checkerpath='./remover%d'%GPARAMS_index 
    if not os.path.exists(Checkerpath):
        os.system("mkdir -p %s"%Checkerpath)
    recordfile=open(Checkerpath+'/remover%d_%d.out'%(GPARAMS_index,GPARAMS.Train_setting.Trainstage),'w')
    EGCMlist=[]
    for i in range(stepnum):
        testset=MSet('TMP')
        testset.mols=tmpset.mols[i*interval:(i+1)*interval]
        num=0
        for imol in testset.mols:
            imol.name="Stage_%d_Mol_%d"%(GPARAMS.Train_setting.Trainstage,num)
            try:
                EGCM=(imol.Cal_EGCM()-GPARAMS.Esoinn_setting.scalemin)/(GPARAMS.Esoinn_setting.scalemax-GPARAMS.Esoinn_setting.scalemin)
                EGCM[~ np.isfinite(EGCM)]=0
                EGCMlist.append(EGCM)
                if GPARAMS.Esoinn_setting.Model.class_id<GPARAMS.Train_setting.Modelnumperpoint:
                    imol.belongto=[i for i in range(GPARAMS.Train_setting.Modelnumperpoint)]
                else:
                    imol.belongto=GPARAMS.Esoinn_setting.Model.find_closest_cluster(GPARAMS.Train_setting.Modelnumperpoint,EGCM)
            except:
                EGCM=imol.Cal_EGCM()
                EGCMlist.append(EGCM)
            num+=1
        ERROR_mols=[]
        maxerr=[]
        if GPARAMS.Compute_setting.Theroylevel=="NN":
            NN_predict,ERROR_mols,maxerr,ERROR_strlist,method=Cal_NN_EFQ(testset)
        else:
            method='No method'
            for j in testset.mols:
                ERROR_mols.append([j,999])
                maxerr.append(999)
        for j in range(len(testset.mols)):
            recordfile.write('step: %8d index :%8d atomnum: %8d error indicator: %12.2f %s\n'%(i,j,len(testset.mols[j].atoms),maxerr[j],method))
            recordfile.flush()
        for j in range(len(ERROR_mols)):
            Queue.put([ERROR_mols[j]])
            unknownlist.append(ERROR_mols[j][0])
        tmpset.mols=unknownlist
        tmpset.Save()
    recordfile.close()
    GPUQueue.put(GPUid)
    return 
