from esoinn import  * 
from copy import deepcopy 
from scipy.spatial.distance import pdist,squareform
class Esoinn_Sys():
    def __init__(self,P,dim,max_edge_age=50,iteration_threshold_list=[200,200],C1=0.001,c2=1.0):
        self.layer1=Esoinn(dim=dim,max_edge_age=max_edge_age,iteration_threshold=iteration_threshold_list[0])
        self.isubgraph_list=[]
        self.instance_list=[]
        self.class_num=0
        self.control_setting=P
        self.dim=dim
        self.max_edge_age=max_edge_age
        self.iteration_threshold=iteration_threshold_list[0]
    
    def train_2layer_esoinn(self,X,if_reset=True,iteration_epochnum=4.0):
        def graph_training(graph,data,if_reset,steps,mse_threshold):
            class_record=[]
            batch_times=steps*0.02
            graph.fit(data,if_reset=if_reset,iteration_times=steps)
            class_record.append(graph.class_id)
            flag=True;time=1
            while flag:
                graph.fit(data,if_reset=False,iteration_times=batch_times)
                class_record.append(graph.class_id)
                time=time+1
                if time> 50:
                    flag=False
                if len(class_record)> 25:
                    tmp=class_record[-5:]
                    avg=np.mean(tmp);mse=np.sum((tmp-avg)**2)/5
                    if mse< 2:
                        flag=False
            return 
        if if_reset==True:
            l1_iteration_times=iteration_epochnum*len(X)
            graph_training(self.layer1,X,l1_iteration_times)
            self.class_num=deepcopy(self.layer1.class_id)
            _,signal_cluster_label,signal_mask=self.layer1.predict(X):
            subgraph_dataset=[[] for m in range(self.layer1.class_id)]
            for i in range(len(x)):
                for j in range(self.class_num):
                    if signal_cluster_label[i] == j:
                        subgraph_dataset[j].append(X[i])
            self.layer.cal_cluster_center()
            self.l1_class_center=np.array(self.layer1.class_center)
            for i in range(self.class_num):
                subgrapph_iteration_times=iteration_epochnum*len(subgraph_dataset[i])
                subgraph=Esoinn(dim=self.dim,max_edge_age=self.max_edge_age,iteration_threshold=self.iteration_threshold_list[1])
                graph_training(subgraph,subgraph_dataset[i],subgraph_iteration_times)
        else:
            dismat=squareform(pdist(self.l1_class_center))
            self.l1_minidis=np.min(dismat,axis=1)
            _,signal_cluster_label,signal_mask=self.layer1.predict(X)
            ref_center=[self.l1_class_center[i] for i in signal_cluster_label]
            ref_dis=[dismat[i] for i in signal_cluster_label]
            X2=[]
            disarray=np.sqrt(np.sum((X-ref_center)**2,axis=1))
            for i in range(len(X)) :
                if disarray[i]>ref_dis[i]:
                    X2.append(X[i])
            l1_iteration_times=iteration_epochnum*len(X2)
            graph_training(self.layer1,if_reset=False,iteration_times=l1_iteration_times)
            class_num=deepcopy(self.layer1.class_id)            
             
                            
         













