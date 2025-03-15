import torch
import torch.nn as nn
import torch.nn.functional as F

#input_dim=1433, hidden_size=256,embedding_size=16 ,
#第一次卷积2708*1433*5*1->2708*704
# 第一次降维2708*704->2708*256
# 第二次降维2708*256->2708*16
class GATLayer(nn.Module):

    def __init__(self, conv_in_features, out_features, alpha=0.2,dataname='Cora'):
        super(GATLayer, self).__init__()
        self.in_features = conv_in_features
        self.out_features = out_features
        self.alpha = alpha
        self.dataname=dataname
        self.W = nn.Parameter(torch.zeros(size=(conv_in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        # 初始化卷积核
        if self.dataname=='Cora':
            self.conv_w = nn.Parameter(torch.zeros(size=(8, 1, 3, 32)))
        elif self.dataname=='Citeseer':
            self.conv_w = nn.Parameter(torch.zeros(size=(5,1,3,32)))#cora用8个
        else:
            self.conv_w = nn.Parameter(torch.zeros(size=(8, 1, 3, 32)))
        nn.init.xavier_uniform_(self.conv_w.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, M, concat=True):
        x=input
        if len(x.shape)==4:#只卷积第一次
            x = F.conv2d(x, weight=self.conv_w, stride=8)
            if self.dataname=='Cora':
                x = torch.reshape(x, (2708, -1))
            elif self.dataname=='Citeseer':
                x = torch.reshape(x, (3327, -1))#cora为2708
            else:
                x=torch.reshape(x, (19717, -1))
        h = torch.mm(x, self.W)
        #下面是引入注意力2708*2708
        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        #乘平均转移状态矩阵
        attn_dense = torch.mul(attn_dense, M)
        attn_dense = self.leakyrelu(attn_dense)  # (N,N)
        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention, h)#注意力系数左乘h,attention其实也很多0
        #h_prime=torch.matmul(adj, h)
        if concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )