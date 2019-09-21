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
a=Neuralnetwork.Esoinn("1AAY")
#a.Load()
#a.Save()
b=load_object('Model.ESOINN')
a.__dict__=b.__dict__
save_object("1AAY.ESOINN",a)
c=load_object("1AAY.ESOINN") 
