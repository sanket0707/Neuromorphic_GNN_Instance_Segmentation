


#read dependancies

from __future__ import division
import os.path as osp

import math
import shutil
import os
import io
import time 
import csv
import sys
import random 
import glob
import pandas as pd
import networkx as nx
import numpy
import numpy as np
import matplotlib.pyplot as plt


##------------------t
import torch
from torch import nn
import torch.nn as nn 
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, BatchNorm1d as BN, ReLU


import torch_scatter
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import max_pool, max_pool_x, graclus, global_mean_pool, GCNConv,  global_mean_pool, SAGEConv
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.transforms import Cartesian
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, download_url
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.nn.pool import radius_graph

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections.abc import Sequence
from collections import Counter
from sklearn.utils import compute_class_weight
from torch import Tensor
try:
    import torch_cluster
except ImportError:
    torch_cluster = None
import time


import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# import script_name_to_call


from read_dataset import train_seq00_p1 as TrainGraphs_seq00



torch.cuda.empty_cache()
torch.backends.cudnn.benchmark=True

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Function and classess
class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=0.5, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
    
# def TRAINING_MODULE(model, number_of_epoch, train_loader, FOLDERTOSAVE): 

def TRAINING_MODULE(model, number_of_epoch, train_loader_part0, train_loader_part1, train_loader_part2, train_loader_part3, train_loader_part4, train_loader_part5, train_loader_part6, train_loader_part7, train_loader_part8, train_loader_part9,train_loader_part10, train_loader_part11, FOLDERTOSAVE):
    model.train()
    nk=10
    lst0= list(np.arange(0, number_of_epoch+1, nk))
    print(lst0)
    lst1= list(np.arange(1, number_of_epoch+1, nk))
    print(lst1)
    lst2= list(np.arange(2, number_of_epoch+1, nk))
    print(lst2)
    lst3= list(np.arange(3, number_of_epoch+1, nk))
    print(lst3)
    lst4= list(np.arange(4, number_of_epoch+1, nk))
    print(lst4)
    lst5= list(np.arange(5, number_of_epoch+1, nk))
    print(lst5)
    lst6= list(np.arange(6, number_of_epoch+1, nk))
    print(lst6)
    lst7= list(np.arange(7, number_of_epoch+1, nk))
    print(lst7)
    lst8= list(np.arange(8, number_of_epoch+1, nk))
    print(lst8)
    lst9= list(np.arange(9, number_of_epoch+1, nk))
    print(lst9)
    lst10= list(np.arange(10, number_of_epoch+1, nk))
    print(lst10)
    i=0
    acc=[]
    epoch_losses = []
    print ("Training will start now")
    for epoch in range(number_of_epoch):
        print("Epoch", epoch)
        epoch_loss = 0
        acc=0
        start=time.time()

        if epoch in lst0:
            train_loader=train_loader_part0
            print("Epoch", epoch ,"train_loader_part0", len(train_loader_part0))
        if epoch in lst1:
            train_loader=train_loader_part2
            print("Epoch", epoch , "train_loader_part1", len(train_loader_part1))

        if epoch in lst2:
            train_loader=train_loader_part4
            print("Epoch", epoch , "train_loader_part2", len(train_loader_part2))

        if epoch in lst3:
            train_loader=train_loader_part6
            print("Epoch", epoch , "train_loader_part3", len(train_loader_part3))

        if epoch in lst4:
            train_loader=train_loader_part8
            print("Epoch", epoch , "train_loader_part4", len(train_loader_part4))


        if epoch in lst5:
            train_loader=train_loader_part0
            print("Epoch", epoch , "train_loader_part5", len(train_loader_part5))

        if epoch in lst6:
            train_loader=train_loader_part2
            print("Epoch", epoch , "train_loader_part6", len(train_loader_part6))

        if epoch in lst7:
            train_loader=train_loader_part4
            print("Epoch", epoch , "train_loader_part7", len(train_loader_part7))

        if epoch in lst8:
            train_loader=train_loader_part6
            print("Epoch", epoch , "train_loader_part8", len(train_loader_part8))

        if epoch in lst9:
            train_loader=train_loader_part8
            print("Epoch", epoch , "train_loader_part9", len(train_loader_part9))

        if epoch in lst10:
            train_loader=train_loader_part11
            print("Epoch", epoch , "train_loader_part10", len(train_loader_part11))



        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
    
            # print ("data.x", data)
   
            end_point = model( data.x, data.pos, data.batch)
            # print("end-point", end_point)
            # print("end-data.y", data.y)

            loss=loss_func(end_point, data.y) 
            # loss=F.nll_loss(end_point, data.y) 

            pred = end_point.max(1)[1]
            acc += (pred.eq(data.y).sum().item())/len(data.y)

            loss.backward()
            optimizer.step() 
            epoch_loss += loss.detach().item()
            i=i+1
        acc /=(i+1)
        epoch_loss /= (i + 1)
        end = time.time()
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), ' Elapsed time: ', end-start, 'Acc', acc)
        epoch_losses.append(epoch_loss)
        torch.save(model.state_dict(), FOLDERTOSAVE+'model_weights.pth')
        torch.save(model, FOLDERTOSAVE+'model.pkl')


        with open(FOLDERTOSAVE+'losses.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows([[loss] for loss in epoch_losses])
            csvFile.close()

        plt.title('cross entropy '+discpt)
        plt.plot(epoch_losses)
        plt.savefig(FOLDERTOSAVE+discpt+str(number_of_epoch)+'epochs.png',dpi=300, bbox_inches='tight')
        plt.savefig(FOLDERTOSAVE+discpt+str(number_of_epoch)+'epochs.pdf', format='pdf', dpi=1200)
# plt.show()
    #     toc()
    return epoch_losses
def TESTING_MODULE(model, test_loader, descrpt):
    model.eval()

    correct = 0
    mov_correct=0
    # T_label=0
    # T_label00=0
    # P_label00=0
    # P_label0=0
    GT=[]
    prediction=[]
    # qqqqqqqq=[]
    id_dataaa=[]
    xd_dataaa=[]
    yd_dataaa=[]
    td_dataaa=[]
    torg=[]
    labeld_dataaa=[]
    for i, data in enumerate(test_loader):
        id_dataaa.append(data.id_data)
        xd_dataaa.append(data.x[:,0])
        yd_dataaa.append(data.x[:,1])
        td_dataaa.append(data.x[:,2])
        labeld_dataaa.append(data.y)
        torg.append(data.t_org)
        
        GT.append(data.y)
        data = data.to(device)
        end_point = model(data.x, data.pos, data.batch)
        loss = loss_func(end_point, data.y)
        pred = end_point.max(1)[1]

        # T_label1=(data.y.eq(1).sum().item())

        # T_label+=T_label1
        # T_label0=(data.y.eq(0).sum().item())
        # T_label00+=T_label0
        # P_label1=(pred.eq(1).sum().item())
        # P_label0+=P_label1
        # P_label0=(pred.eq(0).sum().item())
        # P_label00+=P_label0

        acc = (pred.eq(data.y).sum().item())/len(data.y)
        correct += acc


        prediction.append(pred)
        
    torch.save([numpy.hstack(id_dataaa), numpy.hstack(xd_dataaa), numpy.hstack(yd_dataaa),numpy.hstack(td_dataaa)
                , numpy.hstack(torg),  torch.cat(GT), torch.cat(prediction)], descrpt)
    return [GT, prediction]
def METRICS_MODULE(GT_lbls_,argmax_Y_, csvname ):
    tn, fp, fn, tp=confusion_matrix(torch.cat(GT_lbls_).to('cpu'),torch.cat(argmax_Y_).to('cpu')).ravel()
    Tp_matrix=tp
    Fp_matrix=fp
    Fn_matrix=fn
    Tn_matrix=tn
    print('Tp_matrix_test:',(Tp_matrix))
    print('Fp_matrix_test:' ,(Fp_matrix))
    print('Fn_matrix_test:' ,(Fn_matrix))
    print('Tn_matrix_test:', (Tn_matrix))
    Precision_negative=100*Tn_matrix/(Tn_matrix+Fn_matrix)
    Precision=100*Tp_matrix/(Tp_matrix+Fp_matrix)
    Accuracy=100*((Tp_matrix+Tn_matrix)/(Tp_matrix+Tn_matrix+Fp_matrix+Fn_matrix))
    F1_score=(2*Precision*Accuracy)/(Precision+Accuracy)
    Recall_score=100*Tp_matrix/(Tp_matrix+Fn_matrix)
    Specificity=100*(Tn_matrix)/(Tn_matrix+Fp_matrix)
    print('Precision_negative on the testing set: {:.4f}%'.format(Precision_negative))
    print('Precision on the t0esting set: {:.4f}%'.format(Precision))
    print('Recall on the testing set: {:.4f}%'.format(Recall_score))
    print('F1 score on the testing set: {:.4f}%'.format(F1_score))
    print('Accuracy on the testing set: {:.4f}%'.format(Accuracy))
    print('Specificity on the testing set: {:.4f}%'.format(Specificity))


    with open(csvname, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['Tp_matrix_test:',(Tp_matrix)])
        writer.writerow(['Fp_matrix_test:' ,(Fp_matrix)])
        writer.writerow(['Fn_matrix_test:' ,(Fn_matrix)])
        writer.writerow(['Tn_matrix_test:', (Tn_matrix)])

        writer.writerow(['Precision_negative on the testing set',(Precision_negative)])
        writer.writerow(['Precision on the t0esting set:',(Precision)])
        writer.writerow(['Recall on the testing set: ',(Recall_score)])
        writer.writerow(['F1 score on the testing set:',(F1_score)])
        writer.writerow(['Accuracy on the testing set:',(Accuracy)])
        writer.writerow(['Specificity on the testing set: ',(Specificity)])
    

##---------------------------------------------------------------------------------------------------------------------
# Modeling Parameters and Folders to save
# ---------------------------------------------------------------------------------------------------------------------  
print("INITIALIZATION")
seed_val = int(2)
print("Random Seed ID is: ", seed_val)
random.seed(seed_val)
numpy.random.seed(seed_val)
torch.manual_seed(seed_val)
os.environ['PYTHONHASHSEED'] = str(seed_val)

# device = torch.device(  'cpu')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

learningRate=0.001
number_of_epoch=1000
batch_size=96




# CREATE FOLDERS TO SAVE RESULTS 
discpt='GTNN_3L_trail1_'
FOLDERTOSAVE = 'TrainingResults/'+discpt+'_LR_'+str(learningRate)+"_EPOCH_"+str(number_of_epoch)+'/'
if not os.path.isdir(FOLDERTOSAVE):
    os.makedirs(FOLDERTOSAVE)


##---------------------------------------------------------------------------------------------------------------------
# STAGE B: Network classes
# ---------------------------------------------------------------------------------------------------------------------  
from torch_geometric.nn.conv import PointTransformerConv
import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Identity
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_cluster import fps, knn_graph
from torch_scatter import scatter_max

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import PointTransformerConv
from torch_geometric.nn.pool import knn
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn import global_max_pool



class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], batch_norm=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], batch_norm=False)

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x

class TransitionDown(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels])

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out, _ = scatter_max(x[id_k_neighbor[1]], id_k_neighbor[0],
                               dim_size=id_clusters.size(0), dim=0)

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch




class TransitionDown1(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        print("in_channels", in_channels)
        ratio=0.000000000001
        print("ratio", ratio)

        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels])

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)
        # print("allaa!!!!!!!!!!!!!!!id_clusters", len(id_clusters))

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)
        # print("allaa!!!!!!!!!!!!!!!id_k_neighbor", len(id_k_neighbor))

        # transformation of features through a simple MLP
        x = self.mlp(x)
        # print("allaa!!!!!!!!!!!!!!!x", len(x))

        # Max pool onto each cluster the features from knn in points
        x_out, _ = scatter_max(x[id_k_neighbor[1]], id_k_neighbor[0],
                               dim_size=id_clusters.size(0), dim=0)

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out

        # print("allaa!!!!!!!!!!!!!!!1", (out))
        return out, sub_pos, sub_batch

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]),
            BN(channels[i]) if batch_norm else Identity(), ReLU())
        for i in range(1, len(channels))
    ])
class TransitionUp(torch.nn.Module):
    '''
        Reduce features dimensionnality and interpolate back to higher
        resolution and cardinality
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = MLP([in_channels, out_channels])
        self.mlp = MLP([out_channels, out_channels])

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=3,
                                         batch_x=batch_sub, batch_y=batch)

        x = self.mlp(x) + x_interpolated

        return x



class Net_PointTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]])

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
        )

        # backbone layers
        self.transformers_up = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        self.transition_up = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(0, len(dim_model) - 1):

            # Add Transition Down block followed by a Point Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.k))

            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1],
                                 out_channels=dim_model[i + 1]))

            # Add Transition Up block followed by Point Transformer block
            self.transition_up.append(
                TransitionUp(in_channels=dim_model[i + 1],
                             out_channels=dim_model[i]))

            self.transformers_up.append(
                TransformerBlock(in_channels=dim_model[i],
                                 out_channels=dim_model[i]))

        # summit layers
        self.mlp_summit = MLP([dim_model[-1], dim_model[-1]], batch_norm=False)

        self.transformer_summit = TransformerBlock(
            in_channels=dim_model[-1],
            out_channels=dim_model[-1],
        )

        # class score computation
        self.mlp_output = Seq(Lin(dim_model[0], 64), ReLU(), Lin(64, 64),
                              ReLU(), Lin(64, out_channels))

    def forward(self, x, pos, batch=None):

        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1)).to(pos.get_device())

        out_x = []
        out_pos = []
        out_batch = []

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)

        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)

        # summit
        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_summit(x, pos, edge_index)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.transformers_down)
        for i in range(n):
            x = self.transition_up[-i - 1](x=out_x[-i - 2], x_sub=x,
                                           pos=out_pos[-i - 2],
                                           pos_sub=out_pos[-i - 1],
                                           batch_sub=out_batch[-i - 1],
                                           batch=out_batch[-i - 2])

            edge_index = knn_graph(out_pos[-i - 2], k=self.k,
                                   batch=out_batch[-i - 2])
            x = self.transformers_up[-i - 1](x, out_pos[-i - 2], edge_index)

        # Class score
        out = self.mlp_output(x)

        return F.softmax(out, dim=1)





##---------------------------------------------------------------------------------------------------------------------
# STAGE C: Network parameters
# ---------------------------------------------------------------------------------------------------------------------  

print("STAGE C: BUILDING A NETWORK")

model = Net_PointTransformer(3,10, dim_model=[64, 64, 64], k=16) # 3 is input features 10 is the number of labels


model=model.double()
print("Model Structure ",model)
model=nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
##---------------------------------------------------------------------------------------------------------------------
# STAGE D: TRAINING STAGE
# ---------------------------------------------------------------------------------------------------------------------  
print("STAGE D: TRAINING STAGE - Feedforward")
loss_func=FocalLoss()
train_dataset=TrainGraphs_seq00



## effective training scheme
TrainGraphs_all_part0=TrainGraphs_seq00
TrainGraphs_all_part1=TrainGraphs_seq00
TrainGraphs_all_part2=TrainGraphs_seq00
TrainGraphs_all_part3=TrainGraphs_seq00
TrainGraphs_all_part4=TrainGraphs_seq00

TrainGraphs_all_part5=TrainGraphs_seq00
TrainGraphs_all_part6=TrainGraphs_seq00
TrainGraphs_all_part7=TrainGraphs_seq00
TrainGraphs_all_part8=TrainGraphs_seq00
TrainGraphs_all_part9=TrainGraphs_seq00
TrainGraphs_all_part10=TrainGraphs_seq00
TrainGraphs_all_part11=TrainGraphs_seq00





print("Tranining_part0!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part0))
print("Tranining_part1!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part1))
print("Tranining_part2!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part2))
print("Tranining_part3!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part3))
print("Tranining_part4!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part4))
print("Tranining_part5!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part5))
print("Tranining_part6!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part6))
print("Tranining_part7!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part7))
print("Tranining_part8!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part8))
print("Tranining_part9!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part9))
print("TrainGraphs_all_part10!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part10))
print("TrainGraphs_all_part10!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part11))


###################################################################################
##---------------------type of learning graphs-----------------------------------##
###################################################################################


train_loader_part0 = DataLoader(TrainGraphs_all_part0, batch_size=batch_size, shuffle=True)
train_loader_part1 = DataLoader(TrainGraphs_all_part1, batch_size=batch_size, shuffle=True)
train_loader_part2 = DataLoader(TrainGraphs_all_part2, batch_size=batch_size, shuffle=True)
train_loader_part3 = DataLoader(TrainGraphs_all_part3, batch_size=batch_size, shuffle=True)
train_loader_part4 = DataLoader(TrainGraphs_all_part4, batch_size=batch_size, shuffle=True)
train_loader_part5 = DataLoader(TrainGraphs_all_part5, batch_size=batch_size, shuffle=True)
train_loader_part6 = DataLoader(TrainGraphs_all_part6, batch_size=batch_size, shuffle=True)
train_loader_part7 = DataLoader(TrainGraphs_all_part7, batch_size=batch_size, shuffle=True)
train_loader_part8 = DataLoader(TrainGraphs_all_part8, batch_size=batch_size, shuffle=True)
train_loader_part9 = DataLoader(TrainGraphs_all_part9, batch_size=batch_size, shuffle=True)
train_loader_part10 = DataLoader(TrainGraphs_all_part10, batch_size=batch_size, shuffle=True)
train_loader_part11 = DataLoader(TrainGraphs_all_part11, batch_size=batch_size, shuffle=True)



# breakpoint()
loss_func=FocalLoss()
# model = model.float()

# folder="/home/kunet.ae/100048632/August_models/model9_wMOD_more_3L64NO_GFF/5_Aug2022_TrainingResults_wMOD_3L64NO_GF/5_Aug2022_TrainingResults_wMOD_3L64NO_GF_partialdata_PointTransformer_with connect_LR_0.001_EPOCH_1000batch96/"
# # '/media/yusra/Yusra SSD/from_download/SuccessfulModel_fromCluster/PART2_NEW_MODELS_August19/model4_wMOD_3l64NO_GF/5_Aug2022_TrainingResults_wMOD_3L64NO_GF/connect_LR_0.001_EPOCH_1000batch32/'
# # 'TrainingResults/1_CASE_PointTransfomer'+discpt+'_LR_'+str(learningRate)+"_EPOCH_"+str(number_of_epoch)+'/'

# # folder='/home/kucarst3-dlws/YusraMoseg/newtorch/8_10ms_partialonly1/TrainingResults_partial10ms/1_CASE_PointTransfomerPointTransfomer_LR_0.001_EPOCH_20/'
# # model=torch.load(folder+'model.pkl')
# PATH=folder+'model_weights.pth'
# model.load_state_dict(torch.load(PATH))

# model.eval()
# print("model loaded is done", model)



#-----------------------------------------------------------------------------------------------------------------------------------------
epoch_losses= TRAINING_MODULE(model, number_of_epoch, train_loader_part0, train_loader_part1, train_loader_part2, train_loader_part3, train_loader_part4, train_loader_part5, train_loader_part6, train_loader_part7, train_loader_part8, train_loader_part9, train_loader_part10,train_loader_part11, FOLDERTOSAVE)
#-----------------------------------------------------------------------------------------------------------------------------------------

with open(FOLDERTOSAVE+'losses.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows([[loss] for loss in epoch_losses])
    csvFile.close()




# folder='/home/kucarst3-dlws/YusraMoseg/newtorch/HPC_MoSegModel/TrainingResults/FeedForward_CASE_A_NETWORK_IS_GCN_6LAYERS_LR_0.0001_EPOCH_1/'
# model=torch.load(folder+'model.pkl')
# model.eval()
# print("model loaded is done", model)






# # ---------------------------------------------------------------------------------------------------------------------
# # STAGE E: TESTING STAGE
# # ---------------------------------------------------------------------------------------------------------------------  
print("STAGE E: TESTING STAGE")
