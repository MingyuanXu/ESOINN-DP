from ..Comparm import *
import os
from .Dataer import Check_MSet 
import paramiko as pko 
from .Jobqueue import pbscpustr 

def consumer(Queue):
    import time
    from ..Base import Molnew
    import os
    from TensorMol import MSet 
    print ("Consumer start")
    Newaddedset=MSet('Stage_%d_Newadded'%GPARAMS.Train_setting.Trainstage)
    num=0
    while True:
        ERROR_mols=Queue.get()
        if ERROR_mols==None:
            break
        for i in range(len(ERROR_mols)):
            ERROR_mols[i][0].name="Stage_%d_Mol_%d"%(GPARAMS.Train_setting.Trainstage,num)
            Newaddedset.mols.append(ERROR_mols[i][0])
        num+=1
        if num%2000==0:
            Newaddedset.Save()  
    Dataset=[]
    Newaddedset.mols=Check_MSet(Newaddedset.mols)
    if len(GPARAMS.Esoinn_setting.Model.nodes)!=0 and GPARAMS.Esoinn_setting.Model.class_id > GPARAMS.Train_setting.Modelnumperpoint:
        for i in Newaddedset.mols:
            try:
                Dataset.append(i.EGCM)
            except:
                Dataset.append(i.Cal_EGCM())
        a,b,c,d,signalmask=GPARAMS.Esoinn_setting.Model.predict(Dataset)
        normalmollist=[];edgemollist=[];noisemollist=[]  
        for i in range(len(Newaddedset.mols)):
            if signalmask[i]=='Noise':
                noisemollist.append(Newaddedset.mols[i])
            if signalmask[i]=='Edge':
                edgemollist.append(Newaddedset.mols[i])
            if signalmask[i]=='Normal':
                normalmollist.append(Newaddedset.mols[i])
        print ("Select Newadded set:",len(noisemollist),len(edgemollist),len(normalmollist))
        if len(Newaddedset.mols)>1000:
            edgemollist=random.sample(edgemollist,min(600,len(edgemollist)))
            noisemollist=random.sample(noisemollist,min(200,len(noisemollist)))
            normalmollist=random.sample(normalmollist,min(20,len(normalmollist)))
            Newaddedset.mols=edgemollist+noisemollist+normalmollist 
        print ("After selecting Newadded set:",len(noisemollist),len(edgemollist),len(normalmollist))
    else:
        if len(Newaddedset.mols)>1000:
            Newaddedset.mols=random.sample(Newaddedset.mols,1000)
    Newaddedset.Save()
    return

def calculator(para):
    from ..Comparm import GPARAMS
    mol=para[0]
    input_path=para[1]
    para_path=para[2]
    keywords=para[3]
    ncores=para[4]
    Atomizationlevel=para[5]
    #print (mol,input_path,para_path)

    flag=True

    if 'energy' not in mol.properties.keys():
        if GPARAMS.Compute_setting.Traininglevel=="DFTB3":
            mol.Write_DFTB_input(para_path,False,input_path)
            flag=mol.Cal_DFTB(input_path)
            mol.CalculateAtomization(Atomizationlevel)
        else:
            mol.Write_Gaussian_input(keywords,input_path,ncores,600)
            flag=mol.Cal_Gaussian(input_path)
            mol.CalculateAtomization(Atomizationlevel)
    return (flag,mol)

def parallel_caljob(MSetname,manager,ctrlfile):
    para_path='./'
    if GPARAMS.Compute_setting.Traininglevel=="DFTB+":    
        os.environ["OMP_NUM_THREADS"]=GPARAMS.Compute_setting.Ncoresperthreads
        para_path=GPARAMS.Software_setting.Dftbparapath
    input_path='./'+GPARAMS.Compute_setting.Traininglevel+'/Consumer/'
    if not os.path.exists(input_path):
        os.system("mkdir -p "+input_path)
    TMPSet=MSet(MSetname)
    TMPSet.Load()
    mols=TMPSet.mols
    print ('Nmols in Newaddedset:',len(mols))
    if GPARAMS.Train_setting.Ifcpuwithhelp==True:
        nstage=math.ceil(len(mols)/GPARAMS.Train_setting.framenumperjob)
        print (nstage)
        submollist=[mols[i*GPARAMS.Train_setting.framenumperjob:(i+1)*GPARAMS.Train_setting.framenumperjob] for i in range(nstage)]
        subMSetlist=[MSet(MSetname+'_part%d'%i) for i in range(nstage)]
        subMSetresult=[False for i in range(nstage)]
        for i in range(nstage):
            subMSetlist[i].mols=submollist[i]
            subMSetlist[i].Save()
        trans=pko.Transport((GPARAMS.Train_setting.helpcpunodeip,GPARAMS.Train_setting.helpcpuport))
        trans.connect(username=GPARAMS.Train_setting.helpcpuaccount,password=GPARAMS.Train_setting.helpcpupasswd)
        ssh=pko.SSHClient()
        ssh._transport=trans
        sftp=pko.SFTPClient.from_transport(trans)
        workpath=os.getcwd()
        print (workpath)
        for i in range(nstage):
            subMSetlist[i].mols=submollist[i]
            subMSetlist[i].Save()
            remotepath=GPARAMS.Train_setting.helpcpupath+'/'+MSetname+'/part%d'%i
            srcpath=workpath+'/datasets/%s.pdb'%(MSetname+'_part%d'%i)
            print (" Put pdb file:")
            print (remotepath,srcpath)
            stdin,stdout,stderr=ssh.exec_command('mkdir -p %s/datasets'%remotepath)
            print (stdout.read().decode())
            sftp.put(srcpath,remotepath+'/datasets/%s.pdb'%(MSetname+'_part%d'%i))
            if GPARAMS.Train_setting.cpuqueuetype=='PBS':
                pbsrun=open('pbs.run','w')
                pbsrun.write(pbscpustr%(GPARAMS.Compute_setting.Ncoresperthreads,GPARAMS.Compute_setting.Traininglevel+"_%d"%i))
                pbsrun.write(GPARAMS.Train_setting.helpcpuenv)
                pbsrun.write("Qmcal.py -i %s -d %s> %s.qmout\n"%(ctrlfile,MSetname+'_part%d'%i,MSetname+'_part%d'%i))
                pbsrun.write("rm *.chk\n")
                pbsrun.write("touch finished\n")
                pbsrun.close()
                sftp.put(localpath=workpath+'/pbs.run',remotepath=remotepath+'/pbs.run')
                sftp.put(localpath=workpath+'/'+ctrlfile,remotepath=remotepath+'/'+ctrlfile)
                ssh.exec_command('cd %s && qsub pbs.run'%remotepath)
        t=0
        while False in subMSetresult:
            time.sleep(30)
            t+=30
            for i in range(nstage):
                remotepath=GPARAMS.Train_setting.helpcpupath+'/'+MSetname+'/part%d'%i
                stdin,stdout,stderr=ssh.exec_command("cd %s && ls "%(remotepath))
                tmpstr=stdout.read().decode()
                print ('finished' in tmpstr)
                if 'finished' in tmpstr:
                    subMSetresult[i]=True
            print (t,subMSetresult)
        finishmols=[]
        subMSetlist=[MSet(MSetname+'_part%d'%i) for i in range(nstage)]
        for i in range(nstage):
            srcpath=workpath+'/datasets/%s.pdb'%(MSetname+'_part%d'%i)
            remotepath=GPARAMS.Train_setting.helpcpupath+'/'+MSetname+'/part%d'%i
            os.system('rm %s'%srcpath)
            sftp.get(localpath=srcpath,remotepath=remotepath+'/datasets/'+MSetname+'_part%d.pdb'%i)
            subMSetlist[i].Load()
            finishmols+=subMSetlist[i].mols
        for i in range(len(finishmols)):
            finishmols[i].Cal_EGCM()
        TMPSet.mols=finishmols
        TMPSet.Save()
    else:
        inpathlist=[input_path]*len(mols)
        parapathlist=[para_path]*len(mols)
        corenumperjob=[GPARAMS.Compute_setting.Ncoresperthreads]*len(mols)
        keywordslist=[GPARAMS.Compute_setting.Gaussiankeywords]*len(mols)
        Atomizationlist=[GPARAMS.Compute_setting.Atomizationlevel]*len(mols)
        inputlist=list(zip(mols,inpathlist,parapathlist,keywordslist,corenumperjob,Atomizationlist))
        paracal_pool=manager.Pool(GPARAMS.Compute_setting.Consumerprocessnum)
        results=paracal_pool.map(calculator,inputlist)
        paracal_pool.close()
        paracal_pool.join()
        mollist=[]
        for i in range(len(results)):
            if results[i][0]==True:
                mollist.append(results[i][1])
                mollist[-1].Cal_EGCM()
        TMPSet.mols=mollist
        TMPSet.Save()
        print ("HHHHHHHHHHHHHHHHHHHH")
        print ("HHHHHHHHHHHHHHHHHHHH")
        print (len(TMPSet.mols))
        print ("HHHHHHHHHHHHHHHHHHHH")
        print ("HHHHHHHHHHHHHHHHHHHH")
    return 

