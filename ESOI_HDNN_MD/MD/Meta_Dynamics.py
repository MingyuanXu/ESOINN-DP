import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
from parmed.amber import AmberParm
from parmed.amber import AmberMdcrd
from parmed.amber import Rst7
import pickle
from ..Base import *
import random
from multiprocessing import Queue,Process 
#def BumpEnergy(h,w,distag,dis):
    """

class BumpHolder:
    def __init__(self,CV_num,Max_bump,Height,Width):
        self.max_num=Max_bump
        self.cv_num=cv_num
        self.h_a=Height
        self.w_a=Width
        self.h=None
        self.w=None
        self.Prepare()
        return 
    def Prepare():
        self.Bumpgraph=tf.Graph()
        with self.Bumpgraph.as_default():
            self.xyzs_pl=tf.placeholder(tf.float64,shape=(cv_num,3))
            self.xyzt_pl=tf.placeholder(tf.float64,shape=(cv_num,3))
            self.dis_pl=tf.placeholder(tf.float64,shape=(None,cv_num,1))
            self.h=tf.Variable(self.h_a,dtype=tf.float64)
            self.w=tf.Variable(self.w_a,dtype=tf.float64)
            self.dis=tf.reduce_sum(tf.square(self.xyzs_pl-self.xyzt_pl),1)
            self.BE=BumpEnergy(self.h,self.w,self.dis_pl,self.dis)
            self.BFs=tf.gradients(self.BE,self.xyzs_pl)
            self.BFt=tf.gradients(self.BE,self.xyzt_pl)
            init=tf.global_variables_initializer()
        self.Bumpsess=tf.Session(graph=self.Bumpgraph,config=tf.ConfigProto(allow_soft_placement=True))
    def Bump(dis_,xyzs,xyzt):
        BE,BFs,BFt=self.Bumpsess([self.BE,self.BFs,self.BFt].feed_dict={self.xyzs_pl:xyzs,self.xyzt_pl:xyzt,self.dis_pl:dis_})
        return BE,BFs,BFt 
def BumpEnergy(h,w,dis_,r):
    for i in 

    """
