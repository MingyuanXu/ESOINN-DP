import numpy as np
import argparse
from multiprocessing import Queue,Process,Manager

manager=Manager()
MDresult=manager.dict()
QMqueue=Queue()
EsoinnQueue=Queue()
DataQueue=Queue()

GPUNum=4
Trainlist=[]

Process_list=[]

for times in range(4,7):
    print ('Cycle %d #######################################################################'%times)
    for i in range(4):
        Produc_Process=Process(target=productor,args=(i,topctrl[i],times,MDresult,QMqueue,False))
        Process_list.append(Produc_Process)
        Process_list[-1].start()
    Consume_Process=Process(target=consumer,args=(DFTB_Dict,QMqueue))
    Consume_Process.start()
    for i in range(len(Process_list)):
        Process_list[i].join()
    QMqueue.put(None)
    Consume_Process.join()
    for i in range(len(Process_list)):
        if Process_list[i].is_alive:
            print('%dth process maybe wrong, kill it!!')
            Process_list[i].terminate()
            Process_list[i].join()
    DATA_provider=Process(target=Dataer,args=(DataQueue,GPUNum))
    DATA_provider.start()
    for i in range(GPUNum):
        Trainprocess=Process(target=Trainer,args=(DataQueue,i,True))
        Trainlist.append(Trainprocess)
        Trainlist[-1].start()
    for i in range(GPUNum):
        Trainlist[i].join()
    DATA_provider.join()
    for i in range(len(Process_list)):
        if Trainlist[i].is_alive:
            print('%dth process maybe wrong, kill it!!')
            Trainlist[i].terminate()
            Trainlist[i].join()
