#!/usr/bin/env python
# coding=utf-8
import os,sys
sys.path.append('./Esoinn')
from esoinn import ESoinn,save_object,load_object
import pickle
#from ESOI_HDNN_MD.Neuralnetwork import save_object,load_object
from ESOI_HDNN_MD import *
import json
import numpy as np
a=Neuralnetwork.Esoinn("2L30")
#a.Load()
#a.Save()
b=load_object('MODEL.ESOINN')
for i in b.__dict__.keys():
    if i in a.__dict__.keys():
        a.__dict__[i]=b.__dict__[i]
a.Save()
a.Load()
print (a.nodes)
#save_object("2L30.ESOINN",a)
#c=load_object("2L30.ESOINN") 

