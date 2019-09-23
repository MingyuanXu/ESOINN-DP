import numpy as np                     
from multiprocessing import Queue,Process,Manager,Pool
def productors(index,QMQueue):
    print ('this is an example productor')
    return 

if __name__=="__main__":
    m=Manager()
    QMQueue=m.Queue()
    for i in range(1):
        ProductPool=Pool(len([0]))
        for i in range(len([0])):
            ProductPool.apply_async(productors,(i,QMQueue))
        ProductPool.close()
        ProductPool.join()

