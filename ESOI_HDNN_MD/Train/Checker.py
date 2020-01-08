from ..Comparm import *
import os

def checker(GPARAMS_index=0,Queue=None,GPUQueue=None):
    from ..Base import Find_useable_gpu
    from ..Computemethod import Cal_NN_EFQ 
    print (GPARAMS.Compute_setting.Traininglevel)
    print (GPARAMS.Compute_setting.Theroylevel)
#    os.environ["CUDA_VISIBLE_DEVICES"]=Find_useable_gpu(GPARAMS.Compute_setting.Gpulist)
    GPUid=GPUQueue.get()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(GPUid)
    print (os.environ["CUDA_VISIBLE_DEVICES"])
    tmpset=MSet("Bigset_%d"%GPARAMS_index)
    tmpset.Load()
    interval=5*GPARAMS.Neuralnetwork_setting.Batchsize*5
    stepnum=math.ceil(len(tmpset.mols)/interval)
    Checkerpath='./Checker%d'%GPARAMS_index 
    if not os.path.exists(Checkerpath):
        os.system("mkdir -p %s"%Checkerpath)
    recordfile=open(Checkerpath+'/Checker%d.out'%GPARAMS_index,'w')
    EGCMlist=[]
    for i in range(stepnum):
        testmols=tmpset.mols[i*interval:(i+1)*interval]
        num=0
        for imol in testmols:
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
        if GPARAMS.Compute_setting.Theroylevel=="NN":
            NN_predict,ERROR_mols,AVG_ERR,ERROR_strlist,method=Cal_NN_EFQ(testmols)
            ERROR_mols+=testmols 
        else:
            method='No method'
            for j in testmols:
                ERROR_mols.append([testmols[j],999])
        for j in range(len(ERROR_mols)):
            recordfile.write('step: %8d index :%8d atomnum: %8d error indicator: %12.2f '%(i,j,len(ERROR_mols[j][0].atoms),ERROR_mols[j][1],method))
            recordfile.flush()
            Queue.put(ERROR_mols[j])
    recordfile.close()        
    GPUQueue.put(GPUid)
    return 
        
