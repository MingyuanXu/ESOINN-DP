from ..Comparm import *
import os
from .Dataer import Check_MSet 
import paramiko as pko 
from .Jobqueue import pbscpustr,lsfcpustr 

def consumer(Queue):
    import time
    from ..Base import Molnew
    import os
    from TensorMol import MSet 
    print ("Consumer start")
    Newaddedset=MSet('Stage_%d_Newadded'%GPARAMS.Train_setting.Trainstage)
    num=0
    Error_list=[]
    while True:
        ERROR_mols=Queue.get()
        if ERROR_mols==None:
            break
        for i in range(len(ERROR_mols)):
            ERROR_mols[i][0].name="Stage_%d_Mol_%d_%d"%(GPARAMS.Train_setting.Trainstage,num,i)
            Error_list.append(ERROR_mols[i][1])
            Newaddedset.mols.append(ERROR_mols[i][0])
        num+=1
        if num%2000==0:
            Newaddedset.Save() 
    Error_list=-np.array(Error_list)
    Newaddedset.mols=[Newaddedset.mols[i] for i in np.argsort(Error_list)]

    Dataset=[]
    Newaddedset.mols=Check_MSet(Newaddedset.mols)
    sysnum=(len(GPARAMS.System_setting)+GPARAMS.Compute_setting.Checkernum)
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
        if len(Newaddedset.mols)>GPARAMS.Compute_setting.samplebasenum*sysnum:
            normalnumpersys=math.ceil(GPARAMS.Compute_setting.samplebasenum*0.3)
            edgenumpersys=math.ceil(GPARAMS.Compute_setting.samplebasenum*0.3)
            noisenumpersys=math.ceil(GPARAMS.Compute_setting.samplebasenum*0.3)
            edgemollist=edgemollist[:edgenumpersys*sysnum]
            normalmollist=normalmollist[:normalnumpersys*sysnum]
            noisemolnum=GPARAMS.Compute_setting.samplebasenum*sysnum-len(normalmollist)-len(edgemollist)
            noisemollist=random.sample(noisemollist[:noisemolnum*5],noisemolnum)
            Newaddedset.mols=edgemollist+noisemollist_tmp+normalmollist  
        print ("After selecting Newadded set:",len(noisemollist),len(edgemollist),len(normalmollist))
    else:
        print ("================================")
        print ("samplebasnum&sysnum",GPARAMS.Compute_setting.samplebasenum,sysnum)
        print ("================================")
        if len(Newaddedset.mols)>GPARAMS.Compute_setting.samplebasenum*sysnum:
            Newaddedset.mols=random.sample(Newaddedset.mols,GPARAMS.Compute_setting.samplebasenum*sysnum)
            Newaddedset.mols=Newaddedset.mols[:GPARAMS.Compute_setting.samplebasenum*sysnum]
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
            try:
                flag=mol.Cal_DFTB(input_path)
                mol.CalculateAtomization(Atomizationlevel)
            except:
                flag=False
        else:
            mol.Write_Gaussian_input(keywords,input_path,ncores,600)
            try:
                flag=mol.Cal_Gaussian(input_path)
                mol.CalculateAtomization(Atomizationlevel)
            except:
                flag=False
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
        dftpercpu=math.ceil(len(mols)/GPARAMS.Train_setting.helpcpunum)
        if dftpercpu<GPARAMS.Train_setting.framenumperjob:
            dftpercpu=GPARAMS.Train_setting.framenumperjob 
        nstage=math.ceil(len(mols)/dftpercpu)
        print (nstage)
        submollist=[mols[i*GPARAMS.Train_setting.framenumperjob:(i+1)*GPARAMS.Train_setting.framenumperjob] for i in range(nstage)]
        subMSetlist=[MSet(MSetname+'_part%d'%i) for i in range(nstage)]
        subMSetresult=[False for i in range(nstage)]
        for i in range(nstage):
            subMSetlist[i].mols=submollist[i]
            subMSetlist[i].Save()
        connectflag=True
        connect_num=0
        while connectflag:
            try:
                trans=pko.Transport((GPARAMS.Train_setting.helpcpunodeip,GPARAMS.Train_setting.helpcpuport))
                #trans.banner_timeout=300
                trans.connect(username=GPARAMS.Train_setting.helpcpuaccount,password=GPARAMS.Train_setting.helpcpupasswd)
                connectflag=False
            except Exception as e:
                print (e)
                connect_num+=1
                print (f"{connect_num} th reconnect to {((GPARAMS.Train_setting.helpcpunodeip,GPARAMS.Train_setting.helpcpuport))} for {MSetname}")
                time.sleep(5)
        ssh=pko.SSHClient()
        #ssh.connect(username=GPARAMS.Train_setting.helpcpuaccount,password=GPARAMS.Train_setting.helpcpupasswd,banner_timeout=300,timeout=15)
        ssh._transport=trans
        sftp=pko.SFTPClient.from_transport(trans)
        workpath=os.getcwd()
        print (workpath)
        jobidlist=[]
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
            cpurun=open('cpu.run','w')
            if GPARAMS.Train_setting.cpuqueuetype=='PBS':
                cpurun.write(pbscpustr%(GPARAMS.Compute_setting.Ncoresperthreads,GPARAMS.Compute_setting.Traininglevel+"_%d"%i))
            elif GPARAMS.Train_setting.cpuqueuetype=='LSF':
                cpurun.write(lsfcpustr%(GPARAMS.Compute_setting.Ncoresperthreads,GPARAMS.Compute_setting.Traininglevel+"_%d"%i))
            cpurun.write(GPARAMS.Train_setting.helpcpuenv)
            cpurun.write("rm queue\n")
            cpurun.write('touch started\n')
            cpurun.write("python -u $ESOIHOME/bin/Qmcal.py -i %s -d %s> %s.qmout\n"%(ctrlfile,MSetname+'_part%d'%i,MSetname+'_part%d'%i))
            cpurun.write("rm *.chk started\n")
            cpurun.write("touch finished\n")
            cpurun.close()
            sftp.put(localpath=workpath+'/cpu.run',remotepath=remotepath+'/cpu.run')
            sftp.put(localpath=workpath+'/'+ctrlfile,remotepath=remotepath+'/'+ctrlfile)
            if GPARAMS.Train_setting.cpuqueuetype=='PBS':
                stdin,stdout,stderr=ssh.exec_command('cd %s &&touch queue &&qsub cpu.run'%remotepath)
                jobidlist.append(stdout.read().decode().strip())
                print (jobidlist[-1])
#                stdin,stdout,stderr=ssh.exec_command('cd %s &&ls &&qsub cpu.run'%remotepath)
                #print (stdout.read().decode(),stdout.channel.recv_exit_status(),stderr,stdin,remotepath)
                #print (stdout.read().decode(),stderr,stdin)
            elif GPARAMS.Train_setting.cpuqueuetype=='LSF':
                stdin,stdout,stderr=ssh.exec_command('cd %s && bsub <cpu.run'%remotepath)
                print (stdout.read().decode,stderr,stdin)
        t=0
        while False in subMSetresult:
            time.sleep(30)
            t+=30
            for i in range(nstage):
                remotepath=GPARAMS.Train_setting.helpcpupath+'/'+MSetname+'/part%d'%i
                stdin,stdout,stderr=ssh.exec_command("cd %s && ls "%(remotepath))
                tmpstr=stdout.read().decode()
                if 'finished' in tmpstr:
                    state='finished'
                elif 'started' in tmpstr:
                    state='cal'
                elif 'queue' in tmpstr:
                    state='queue'
                if 'finished' in tmpstr:
                    subMSetresult[i]=True
                stdin,stdout,stderr=ssh.exec_command('qstat')
                tmpstr=stdout.read().decode()
                if jobidlist[i] not in tmpstr and state=='queue':
                    stdin,stdout,stderr=ssh.exec_command('cd %s && qsub cpu.run'%remotepath)
                    newid=stdout.read().decode().strip()
                    jobidlist[i]=newid 
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
        corenumperjob=[math.ceil(GPARAMS.Compute_setting.Ncoresperthreads/GPARAMS.Compute_setting.Consumerprocessnum)]*len(mols)
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

