import numpy as np
import tensorflow as tf

class BumpHolder():
    def __init__(self,CV_num,Max_bump,Height,Width):
        self.max_num=Max_bump
        self.cv_num=CV_num
        self.h_a=Height
        self.w_a=Width
        self.h=None
        self.w=None
        self.Prepare()
        return

    def Prepare(self):
        self.Bumpgraph=tf.Graph()
        with self.Bumpgraph.as_default():
            self.xyzs_pl=tf.placeholder(tf.float64,shape=(self.cv_num,3))
            self.xyzt_pl=tf.placeholder(tf.float64,shape=(self.cv_num,3))
            self.dis_pl=tf.placeholder(tf.float64,shape=(None,self.cv_num,1))
            self.h=tf.Variable(self.h_a,dtype=tf.float64)
            self.w=tf.Variable(self.w_a,dtype=tf.float64)
            self.dis=tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(self.xyzs_pl-self.xyzt_pl),1)),(-1,self.cv_num,1))
            self.frame_diff=tf.square(self.dis_pl-self.dis)
            self.Bumpenergy=-1*self.h*tf.exp(-0.5*self.frame_diff/tf.square(self.w))
            self.Bumpforce1=tf.gradients(self.Bumpenergy,self.xyzs_pl)
            self.Bumpforce2=tf.gradients(self.Bumpenergy,self.xyzt_pl)
            #self.Bumpenergy=tf.
            init=tf.global_variables_initializer()
        self.Bumpsess=tf.Session(graph=self.Bumpgraph,config=tf.ConfigProto(allow_soft_placement=True))
        self.Bumpsess.run(init)

    def Cal_Bumpforce(self,xyzs,xyzt,dis_pl):
        BF1=self.Bumpsess.run(self.Bumpforce1,{self.xyzs_pl:xyzs,self.xyzt_pl:xyzt,self.dis_pl:dis_pl})
        return BF1

coords=np.array([[0.0,1.0,1.0],[2.0,1.0,1.0],[1.0,2.0,1.0],[0.0,2.0,0.0]])
sindex=np.array([0,1])
tindex=np.array([2,3])
xyzs=coords[sindex]
xyzt=coords[tindex]
print (coords,xyzs,xyzt)
dis_pl=np.array([[[0],[0]],[[1],[1]]])
bump=BumpHolder(2,2,1,1)
bump.Prepare()
BF1=bump.Cal_Bumpforce(xyzs,xyzt,dis_pl)
print (BF1)

