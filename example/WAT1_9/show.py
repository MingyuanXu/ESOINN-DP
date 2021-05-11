import numpy as np                     
from ESOI_HDNN_MD.Computemethod import Qmmm
from ESOI_HDNN_MD.Comparm import GPARAMS
from ESOI_HDNN_MD.Base.Info import List2str
from ESOI_HDNN_MD import UpdateGPARAMS,LoadModel,Added_MSet
from ESOI_HDNN_MD.Train import productor,consumer,esoinner,trainer,dataer,parallel_caljob,get_best_struc 
import os
import argparse as arg
from multiprocessing import Queue,Process,Manager,Pool
import time
import dash 
import dash_core_components as doc
import dash_html_components as html
import plotly.graph_objs as go 
from dash.dependencies import Input,Output
import pandas as pd
import pickle 
import dash_bio as dashbio
import dash_bio_utils.xyz_reader as xyz_reader
from TensorMol import *


app=dash.Dash(__name__,meta_tags=[{"name":"viewport","content":"width=device-width"}])
server=app.server 
parser=arg.ArgumentParser(description='Grep qm area from an Amber MDcrd trajory to make training dataset!')
parser.add_argument('-i','--input')
args=parser.parse_args()
jsonfile=args.input

UpdateGPARAMS(jsonfile)
LoadModel(ifhdnn=False)
Refset=MSet(GPARAMS.Compute_setting.Traininglevel)
Refset.Load()
try:
    representid=GPARAMS.Esoinn_setting.Model.represent_strucid
except:
    signallist=np.array([i.EGCM for  i in Refset.mols])
    signallist=(signallist-GPARAMS.Esoinn_setting.scalemin)/(GPARAMS.Esoinn_setting.scalemax-GPARAMS.Esoinn_setting.scalemin)
    signallist[~np.isfinite(signallist)]=0
    representid=GPARAMS.Esoinn_setting.Model.select_represent_struc(signallist)
    GPARAMS.Esoinn_setting.Model.Save()
representstruc=[]
for  i in range(len(representid)):
    xyzstr=Refset.mols[representid[i]].transstr()
    representstruc.append(xyzstr)
speckdata=xyz_reader.read_xyz(datapath_or_datastring=representstruc[0],is_datafile=False)
moddevlist=[[[] for j in range(GPARAMS.Train_setting.Trainstage)] for i in range(len(GPARAMS.MD_setting))]
templist=[[[] for j in range(GPARAMS.Train_setting.Trainstage)] for i in range(len(GPARAMS.MD_setting))]
timelist=[[[] for j in range(GPARAMS.Train_setting.Trainstage)] for i in range(len(GPARAMS.MD_setting))]
graphlist=[] 
modeldev=[[] for j in range(GPARAMS.Train_setting.Trainstage)]
for i in range(len(GPARAMS.MD_setting)):
    
    timemask=0
    moddevdatlist=[]
    tempdatlist=[]
    for j in range(GPARAMS.Train_setting.Trainstage):
        MDout='%s/%s_%d'%(GPARAMS.MD_setting[i].Name,GPARAMS.MD_setting[i].Name,j)
        MDoutfile=MDout+'.mdout'
        if os.path.exists(MDoutfile):
            file=open(MDoutfile,'r')
            for eachline in file:
                var=eachline.split()
                time=float(var[4].strip('(fs)'));err=float(var[15]);temp=float(var[12])
                timelist[i][j].append(time+timemask)
                templist[i][j].append(temp)
                moddevlist[i][j].append (math.sqrt(err))
            timemask=timelist[i][j][-1]
        else:
            print ("%s is Lost! the summary program will ignore the result in this stage!")
        a=np.array([[timelist[i][j][k],moddevlist[i][j][k]] for k in range(len(moddevlist[i][j])) if k%50==0]).T 
        modeldev[j]+=[[timelist[i][j][k],moddevlist[i][j][k]] for k in range(len(moddevlist[i][j]))]
        b=np.array([[timelist[i][j][k],templist[i][j][k]] for k in range(len(templist[i][j])) if k%50==0]).T 
        moddevdat=go.Scatter(
            x=a[0],
            y=a[1],
            text="Stage: %d"%j,
            mode='markers+lines',
            opacity=0.7,
            marker={'size':3},
            name="Stage: %d"%j
        )
        moddevdatlist.append(moddevdat)
        moddevlayout=go.Layout(
            autosize=True,
            xaxis={"title":"Time (fs)"},
            yaxis={"title":"Model deviation"},
            margin={"l":60,"b":40,'t':40,'r':60},
            legend={"x":0.8,"y":1},
            hovermode='closest',
            #plot_bgcolor="#F9F9F9",
            #paper_bgcolor="#F9F9F9"
        )
        tempdat=go.Scatter(
            x=b[0][1:],
            y=b[1][1:],
            text="Stage: %d"%j,
            mode='markers+lines',
            opacity=0.7,
            marker={'size':3},
            name="Stage: %d"%j
        )
        tempdatlist.append(tempdat)
        templayout=go.Layout(
            xaxis={"title":"Time (fs)"},
            yaxis={"title":"Temperature (K)"},
            margin={"l":40,"b":40,'t':40,'r':40},
            legend={"x":0.8,"y":1},
            #paper_bgcolor="#F9F9F9",
            hovermode='closest'
        )
        #templayout.yaxis.range=[100,600]
    moddevgraph=doc.Graph(id=MDout,
                    figure={'data':moddevdatlist,"layout":moddevlayout}
                   )
    tempdevgraph=doc.Graph(
                    id=MDout+'_temp',
                    figure={'data':tempdatlist,"layout":templayout} 
                    )
    graphlist.append([moddevgraph,tempdevgraph])
for j in range(GPARAMS.Train_setting.Trainstage):
    np.savetxt('stage%d.modeldev'%j,np.array(modeldev[j]))
Model_F_result=[]
Model_E_result=[]
Model_D_result=[]
LR_list=[]
if os.path.exists(GPARAMS.Compute_setting.Traininglevel):
    for i in range(GPARAMS.Train_setting.Trainstage):
        path='%s/Stage%d'%(GPARAMS.Compute_setting.Traininglevel,i)
        if os.path.exists(path):
            LR_list.append([])
            Model_E_result.append([])
            Model_F_result.append([])
            Model_D_result.append([])
            filelist=os.listdir(path)
            modelnum=len(filelist) 
            for j in range(modelnum):
                Model_E_result[-1].append([])
                Model_F_result[-1].append([])
                Model_D_result[-1].append([])
                LR_list[-1].append([])
                file=open(path+'/'+filelist[j],'r')
                Mask='Dipole'
                for eachline in file:
                    if 'step' in eachline and 'Test' in eachline:
                        var=eachline.split()
                        step=int(var[1]);learningrate=float(var[3]);trainlosse=float(var[11]);trainlossf=float(var[13].strip(','));
                        trainlossd=float(var[15]);testlosse=float(var[21]);testlossf=float(var[23].strip(','));testlossd=float(var[25])
                        LR_list[i][j].append([step,learningrate])
                        if Mask=='Dipole':
                            Model_D_result[i][j].append([step,trainlossd,testlossd])
                        elif Mask=='EG':
                            Model_E_result[i][j].append([step,trainlosse,testlosse])
                            Model_F_result[i][j].append([step,trainlossf,testlossf])
                    if 'Switching' in eachline:
                        Mask='EG'
        else:
            print ("%s is lost"%path)
else:
    print ("Training result of ESOI-HDNN model has lost")
if os.path.exists("NNstruc.record"):
    df=pd.read_csv("NNstruc.record")
    df=pd.DataFrame(df.round(6),dtype=str)
def generate_table(dataframe):
    fig=doc.Graph(id='table',
                  figure={'data':[go.Table(
                                    header=dict(values=['<b>%s</b>'%i for i in list(df.columns)],fill_color='darkslategray',align='center',font=dict(size=14,color='white')),
                                    cells=dict(values=[df['Data num'],df['Batch num for EG'],df['Loss E'],df['Loss F'],df['Batch num for Dipole'],df['Loss D'],df['Net Structure']],\
                                               fill_color='lightgrey',align='center',font=dict(size=14))
                                )],
                          'layout':go.Layout(margin=dict(b=20,l=100,r=100,t=20))
                        }
                )
    return [fig]

if not os.path.exists("ESOI-Layer.History"):
    from copy import deepcopy
    from sklearn import manifold
    from sklearn.metrics import euclidean_distances 
    #total_data=list(deepcopy(GPARAMS.Esoinn_setting.Model.learn_history_node))
    total_pt=[]
    total_data=[]
    for i in range(len(GPARAMS.Esoinn_setting.Model.learn_history_node)):
        total_pt.append([])
        total_pt[-1].append(len(total_data))
        total_data=total_data+list(GPARAMS.Esoinn_setting.Model.learn_history_node[i])
        total_pt[-1].append(len(total_data))
    total_pt.append([])
    total_pt[-1].append(len(total_data))
    total_data+=list(GPARAMS.Esoinn_setting.Model.nodes)
    total_pt[-1].append(len(total_data))
    similarities=euclidean_distances(np.array(total_data))
    mds=manifold.MDS(n_components=2,max_iter=500,eps=1e-7,dissimilarity="precomputed",n_jobs=GPARAMS.Compute_setting.Ncoresperthreads)
    pos=mds.fit(similarities).embedding_
    total_2D_data=[]
    for i in range(len(GPARAMS.Esoinn_setting.Model.learn_history_node)):
        total_2D_data.append([])
        total_2D_data[-1]=pos[total_pt[i][0]:total_pt[i][1]]
    total_2D_data.append([])
    total_2D_data[-1]=pos[total_pt[-1][0]:total_pt[-1][1]]
    with open("ESOI-Layer.History",'wb') as file:
        pickle.dump(total_2D_data,file)
else:
    with open("ESOI-Layer.History",'rb') as file:
        total_2D_data=pickle.load(file)
    
app.layout=html.Div(children=[
    html.Div(children=[
        html.Img(src=app.get_asset_url('ESOI-HDNN-MD-Logo.png'),style={"width":"60%","margin":"auto","align-items":"center","display":"flex","justify":"center"}),
        html.P(children=' ',style={"text-align":"center","height":"20px"})
    ])]+\
    [
        html.Div(
        html.Div(
            children=[html.Img(src=app.get_asset_url('ESOI-HDNN-MD-introduction2.png'),id="Simple introduction",style={"width":"100%","text-align":"center"})],
            className="pretty_container two-thirds columns"
        ),className="raw flex-display"
        )
    ]+\
    [
        html.P(children=' ',style={"text-align":"center","height":"20px"}),
        html.H5(children="------- ESOI-Layer Structure -------",style={"margin":"auto","text-align":"center"}),
        html.P(children=' ',style={"text-align":"center","height":"20px"})
    ]+\
    [   
        html.Div([
            html.Div([doc.Graph(id="ESOI-Layer")],className='pretty_container eight columns'),
            html.Div([
                dashbio.Speck(id='my-dashbio-speck',
                              view={'resolution':600,'zoom':0.2,'atomScale':0.2,'bonds':True},\
                              data=speckdata,presetView='stickball')
            ],className='pretty_container four columns')
        ],className="raw flex-display"),
        html.Div([doc.Slider(
                    id="ESOI-Layer Training Stage",
                    min=0,
                    max=len(GPARAMS.Esoinn_setting.Model.learn_history_edge),
                    value=1,
                    marks={str(i):str(i) for i in range(len(GPARAMS.Esoinn_setting.Model.learn_history_edge)+1)},
                    step=None
        )],style={"width":"100%","margin":"auto","height":"80px"})
    ]+\
    [
        html.P(children=' ',style={"text-align":"center","height":"20px"}),
        html.H5(children="------- Meta-networks Training Result -------",style={"margin":"auto","text-align":"center"}),
        html.P(children=' ',style={"text-align":"center","height":"20px"})
    ]+\
    [
        html.Div([
            html.Div([doc.Graph(id="E")],className='pretty_container four columns'),
            html.Div([doc.Graph(id="F")],className='pretty_container four columns'),
            html.Div([doc.Graph(id="D")],className='pretty_container four columns')    
            ],className="raw flex-display"),
        html.Div([doc.Slider(
                    id="Training Stage",
                    min=0,
                    max=GPARAMS.Train_setting.Trainstage-1,
                    value=1,
                    marks={str(i):str(i) for  i in range(GPARAMS.Train_setting.Trainstage)},
                    step=None 
        )],style={"width":"100%","margin":"auto","height":"80px"})
    ]+\
    [
        html.H5(children="------- Optimization of Unit Net structure with Genetic Algorithm -------\n",style={"margin":"auto","text-align":"center"}),
        html.Div(children=generate_table(df),className='pretty_container twelve columns'),
        html.H5(children="------- MD Simulation in ESOI-HDNN Training Process -------",\
                style={"margin":"auto","text-align":"center"})
    ]+\
    [
        html.Div(children=[
            html.H6(children="%s"%GPARAMS.MD_setting[i].Name,className="label",style={"margin":"auto","text-align":"center"}),
            html.Div(children=[
                html.Div(children=graphlist[i][0],className="pretty_container six columns"),
                html.Div(children=graphlist[i][1],className="pretty_container six columns")
                        ],
            className="raw flex-display"
            )
        ])
        for i in range(len(GPARAMS.MD_setting))
    ],
    style={"margin":"auto","color":"grey"}
)
@app.callback(Output('ESOI-Layer','figure'),\
             [Input("ESOI-Layer Training Stage","value")])
def Update_ESOI_layer_struc(select_stage):
    id=int(select_stage)
    edgex=[]
    edgey=[]
    adjacency=np.zeros(len(total_2D_data[id]))
    if id <len(GPARAMS.Esoinn_setting.Model.learn_history_edge):
        nodelist=go.Scatter(x=total_2D_data[id][:,0],y=total_2D_data[id][:,1],text="Stage%d"%id,name="Stage%d"%id,\
                    mode="markers",hoverinfo='text',
                    marker=dict(showscale=True,colorscale='YlGnBu',reversescale=True,color=[],size=10,\
                         colorbar=dict(thickness=15,title='Node Connections',xanchor='right',titleside='right'),\
                    line_width=2))
        for j in GPARAMS.Esoinn_setting.Model.learn_history_edge[id]:
            total_2D_data[id]=np.array(total_2D_data[id])
            edgex+=list(total_2D_data[id][j,0])+[None]
            edgey+=list(total_2D_data[id][j,1])+[None]
            adjacency[j[0]]+=1
    else:
        nodelist=go.Scatter(x=total_2D_data[id][:,0],y=total_2D_data[id][:,1],text="Stage%d"%id,name="Stage%d"%id,\
                    mode="markers",hoverinfo='text',
                    marker=dict(size=10,\
                    line_width=2))
        for j in list(GPARAMS.Esoinn_setting.Model.adjacent_mat.keys()):
            edgex+=list(total_2D_data[id][j,0])+[None]
            edgey+=list(total_2D_data[id][j,1])+[None]
            adjacency[j[0]]+=1
    edgelist=go.Scatter(x=edgex,y=edgey,line=dict(width=0.5,color='#888'),hoverinfo='none',mode='lines')
    node_text=['#with %d connections'%adjacency[j] for j in range(len(adjacency))]

    if id <len(GPARAMS.Esoinn_setting.Model.learn_history_edge):
        nodelist.marker.color=adjacency
    else:
        sizelist=np.array(GPARAMS.Esoinn_setting.Model.density)
        maxsize=np.max(sizelist)
        minsize=np.min(sizelist)
        sizelist=(sizelist-minsize)/(maxsize-minsize)*90+10
        nodelist.marker.size=sizelist 
        node_text=['Density %.2f'%GPARAMS.Esoinn_setting.Model.density[j] for j in range(len(sizelist))]
    nodelist.text=node_text
    x0=np.array(total_2D_data[id][:,0])
    y0=np.array(total_2D_data[id][:,1])

    histogram2d=go.Histogram2d(x=x0,\
                               y=y0,\
                               histnorm='probability',
                               colorscale='YlGnBu',
                               nbinsx=50,
                               nbinsy=50)
    ESOILayout=go.Layout(showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                autosize=False,
                height=600,clickmode='event+select')
    ESOILayer={"data":[histogram2d,edgelist,nodelist],"layout":ESOILayout}
    return ESOILayer 

@app.callback(Output('my-dashbio-speck','data'),\
             [Input("ESOI-Layer","clickData"),Input("ESOI-Layer Training Stage","value")])
def show_struc(click_data,stage):
    if stage==len(GPARAMS.Esoinn_setting.Model.learn_history_edge):
        speckdata=xyz_reader.read_xyz(datapath_or_datastring=representstruc[click_data['points'][0]['pointIndex']],is_datafile=False)
    else: 
        speckdata=xyz_reader.read_xyz(datapath_or_datastring=representstruc[0],is_datafile=False)
    return  speckdata

@app.callback([Output('E','figure'),Output('F','figure'),Output('D','figure')],\
             [Input("Training Stage","value")])
def Update_Trainingresult(select_stage):
    id=int(select_stage)
    Etraces=[];Ftraces=[];Dtraces=[];LRtraces=[]
    modelnum=len(LR_list[id])
    for i in range(modelnum):
        tmpdat=np.array([Model_E_result[id][i][j] for j in range(len(Model_E_result[id][i])) if j%5==0]).T
        Etraces.append(
            go.Scatter(
                x=tmpdat[0],
                y=tmpdat[1],
                text="Model%d:Train"%i,
                name="Model%d:Train"%i,
                mode='markers+lines',
                marker={"size":3},
                opacity=0.7
            )
        )
        Etraces.append(
            go.Scatter(
                x=tmpdat[0],
                y=tmpdat[2],
                text="Model%d:Test"%i,
                name="Model%d:Test"%i,
                mode='markers+lines',
                marker={"size":3},
                opacity=0.7
            )
        )
        tmpdat=np.array([Model_F_result[id][i][j] for j in range(len(Model_F_result[id][i])) if j%5==0]).T
        Ftraces.append(
            go.Scatter(
                x=tmpdat[0],
                y=tmpdat[1],
                text="Model%d:Train"%i,
                name="Model%d:Train"%i,
                mode='markers+lines',
                marker={"size":3},
                opacity=0.7
            )
        )
        Ftraces.append(
            go.Scatter(
                x=tmpdat[0],
                y=tmpdat[2],
                text="Model%d:Test"%i,
                name="Model%d:Test"%i,
                mode='markers+lines',
                marker={"size":3},
                opacity=0.7
            )
        )
        tmpdat=np.array([Model_D_result[id][i][j] for j in range(len(Model_D_result[id][i])) if j%5==0]).T
        Dtraces.append(
            go.Scatter(
                x=tmpdat[0],
                y=tmpdat[2],
                text="Model%d:Train"%i,
                name="Model%d:Train"%i,
                mode='markers+lines',
                marker={"size":3},
                opacity=0.7
            )
        )
        Dtraces.append(
            go.Scatter(
                x=tmpdat[0],
                y=tmpdat[2],
                text="Model%d:Test"%i,
                name="Model%d:Test"%i,
                mode='markers+lines',
                marker={"size":3},
                opacity=0.7
            )
        )

    Elayout=go.Layout(xaxis={"title":"Epoch Num"},yaxis={"title":"E Loss","range":[0,2]},margin={"l":40,"b":40,'t':10,'r':10},legend={"x":0.75,"y":1},hovermode='closest')
    Flayout=go.Layout(xaxis={"title":"Epoch Num"},yaxis={"title":"F Loss","range":[0,2]},margin={"l":40,"b":40,'t':10,'r':10},legend={"x":0.75,"y":1},hovermode='closest')
    Dlayout=go.Layout(xaxis={"title":"Epoch Num"},yaxis={"title":"D Loss","range":[0,2]},margin={"l":40,"b":40,'t':10,'r':10},legend={"x":0.75,"y":1},hovermode='closest')
    Efigure={"data":Etraces,'layout':Elayout}
    Ffigure={"data":Ftraces,'layout':Flayout}
    Dfigure={"data":Dtraces,'layout':Dlayout}
    return Efigure,Ffigure,Dfigure    

if __name__=='__main__':
    app.run_server(debug=True)

