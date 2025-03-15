import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import GATLayer
#hidden_size=256，embedding_size=16
class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha,dataname):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        if dataname=='Cora':
            self.conv_num_features=1408
        elif dataname=='Citeseer':
            self.conv_num_features = 2295
        elif dataname == 'Wiki':
            self.conv_num_features=3090
        else:
            self.conv_num_features = 472
        self.conv1 = GATLayer(self.conv_num_features, hidden_size, alpha,dataname)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha,dataname)
        #self.conv3 = GATLayer(embedding_size, 16, alpha)
    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
