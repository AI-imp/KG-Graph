import numpy as np
import torch
from sklearn.preprocessing import normalize

from torch_geometric.datasets import Planetoid
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

def get_dataset(dataset):
    datasets = Planetoid('./dataset', dataset)
    return datasets

def data_preprocessing(dataset):
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj
    dataset.adj += torch.eye(dataset.x.shape[0])
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)
    return dataset

def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t=2#只聚合两层邻居的状态
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset='pubmed'):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open(r"E:\python\learningproject\gcn\SGAE_NFC\dataset\{}\raw/ind.{}.{}".format(dataset.capitalize(),dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj

def reorgonize_features(features, adj,nd_avg):
    feature=features
    features=feature.detach().numpy()#tensor开始训练就有梯度信息
    nd = np.array(adj.sum(1))#每个节点的度
    #print('==== the number of the neighbors ====', nd_avg)  # 选择邻居n-1
    adj = adj.toarray()  ##稀疏转密集
    adj0 = np.zeros(adj.shape)
    nd0 = nd[0]
    addr = np.where(adj[0] > 0)
    if int(nd0) > nd_avg:
        # ***** choose by the features similarity *******
        aa = features[addr[0]]
        aa0 = np.vstack((features[0], aa))
        bb = cosine_similarity(aa0)
        cc = bb[0][1:]
        cc_addr1 = np.argpartition(cc, -int(nd_avg))[-int(nd_avg):]  # get the top nd_avg neighbor

        addr1 = addr[0][cc_addr1]
        fea_mer0 = np.vstack((features[addr1], features[0]))
        fea_mer0 = np.expand_dims(fea_mer0, 0)  # (1, 5, 1433)
        new_features = fea_mer0
        adj0[0][addr1] = 1
    else:
        gap = nd_avg - int(nd[0]) + 1  # gap+1, consider merge node self features
        # gap+2, consider merge node self and the left added neighbors
        fea_mer1 = features[addr]
        for k in range(gap):
            fea_mer1 = np.concatenate([features[0].reshape(1, features.shape[1]), fea_mer1])
        fea_mer1 = np.expand_dims(fea_mer1, 0)
        new_features = fea_mer1
    for i in range(1, len(nd)):
        addr0 = np.where(adj[i] == 1)  # find all one
        if int(nd[i]) >= nd_avg:
            aa = features[addr0[0]]
            aa0 = np.vstack((features[i], aa))
            bb = cosine_similarity(aa0)
            cc = bb[0][1:]
            cc_addr1 = np.argpartition(cc, -int(nd_avg))[0:nd_avg]
            addr1 = addr0[0][cc_addr1]
            fea_mer3 = np.vstack((features[i], features[addr1]))  # (5, 1433)
            fea_mer3 = np.expand_dims(fea_mer3, 0)  # (1, 5, 1433)
            new_features = np.concatenate((new_features, fea_mer3))

            adj0[i][addr1] = 1
        else:
            gap = nd_avg - int(nd[i]) + 1
            fea_mer4 = features[addr0]
            for k in range(gap):
                fea_mer4 = np.concatenate([features[i].reshape(1, features.shape[1]), fea_mer4], 0)
            fea_mer4 = np.expand_dims(fea_mer4, 0)
            new_features = np.concatenate((new_features, fea_mer4))
    return new_features

def reorgonize_features_wiki(features, adj,nd_avg):
    feature=features
    features=feature.detach().numpy()#tensor开始训练就有梯度信息
    nd = np.array(adj.sum(1))#每个节点的度
    #print('==== the number of the neighbors ====', nd_avg)  # 选择邻居n-1
    #adj = adj.toarray()  ##稀疏转密集
    adj0 = np.zeros(adj.shape)
    nd0 = nd[0]
    addr = np.where(adj[0] > 0)
    if int(nd0) > nd_avg:
        # ***** choose by the features similarity *******
        aa = features[addr[0]]
        aa0 = np.vstack((features[0], aa))
        bb = cosine_similarity(aa0)
        cc = bb[0][1:]
        cc_addr1 = np.argpartition(cc, -int(nd_avg))[-int(nd_avg):]  # get the top nd_avg neighbor

        addr1 = addr[0][cc_addr1]
        fea_mer0 = np.vstack((features[addr1], features[0]))
        fea_mer0 = np.expand_dims(fea_mer0, 0)  # (1, 5, 1433)
        new_features = fea_mer0
        adj0[0][addr1] = 1
    else:
        gap = nd_avg - int(nd[0]) + 1  # gap+1, consider merge node self features
        # gap+2, consider merge node self and the left added neighbors
        fea_mer1 = features[addr]
        for k in range(gap):
            fea_mer1 = np.concatenate([features[0].reshape(1, features.shape[1]), fea_mer1])
        fea_mer1 = np.expand_dims(fea_mer1, 0)
        new_features = fea_mer1
    for i in range(1, len(nd)):
        addr0 = np.where(adj[i] == 1)  # find all one
        if int(nd[i]) >= nd_avg:
            aa = features[addr0[0]]
            aa0 = np.vstack((features[i], aa))
            bb = cosine_similarity(aa0)
            cc = bb[0][1:]
            cc_addr1 = np.argpartition(cc, -int(nd_avg))[0:nd_avg]
            addr1 = addr0[0][cc_addr1]
            fea_mer3 = np.vstack((features[i], features[addr1]))  # (5, 1433)
            fea_mer3 = np.expand_dims(fea_mer3, 0)  # (1, 5, 1433)
            new_features = np.concatenate((new_features, fea_mer3))

            adj0[i][addr1] = 1
        else:
            gap = nd_avg - int(nd[i]) + 1
            fea_mer4 = features[addr0]
            for k in range(gap):
                fea_mer4 = np.concatenate([features[i].reshape(1, features.shape[1]), fea_mer4], 0)
            fea_mer4 = np.expand_dims(fea_mer4, 0)
            new_features = np.concatenate((new_features, fea_mer4))
    return new_features