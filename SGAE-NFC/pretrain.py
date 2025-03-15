import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
import utils
from model import GAT
from evaluation import eva

def pretrain(dataset):
    model = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
        dataname=args.name
    ).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)
    # data and label
    x = torch.Tensor(dataset.x).to(device)
    if args.name=='Pubmed':
        x = torch.where(x > 0, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
    #处理使用nfc聚合邻居特征
    adj0 = utils.load_data(args.name[0].lower() + args.name[1:])
    x=utils.reorgonize_features(x,adj0,args.nd_avg)#输出为(2708*5*1433)
    x = np.expand_dims(x,1)
    x=torch.tensor(x).to(torch.float32)
    y = dataset.y.cpu().numpy()#用上了标签
    for epoch in range(args.max_epoch):
        model.train()
        A_pred, z = model(x, adj, M)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))#将x训练到能预测边的真实分布
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():#这一次用来评估
            _, z = model(x, adj, M)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )
            acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)#评估的时候用上了y真实，但实际只需训练结果save好即可
        if epoch % 5 == 0 or nmi>0.42:
            torch.save(
                model.state_dict(
                ), f"pretrain/predaegc_{args.name}_{epoch}.pkl"
            )

if __name__ == "__main__":
    # 描述性信息
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 命令行后面添加的参数
    parser.add_argument("--name", type=str, default="Citeseer")
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_clusters", default=6, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--embedding_size", default=64, type=int)
    parser.add_argument("--weight_decay", type=int, default=5e-3)
    parser.add_argument("--nd_avg", type=int, default=5)
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
    )
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    # 是否使用cuda，
    print("use cuda: {}".format(args.cuda))
    # 选择使用cuda还是cpu。
    device = torch.device("cuda" if args.cuda else "cpu")
    # 数据集
    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]

    if args.name == "Citeseer":
        args.lr = 0.005
        args.k = None
        args.n_clusters = 6
    elif args.name == "Cora":
        args.lr = 0.005
        args.k = None
        args.n_clusters = 7
    elif args.name == "Pubmed":
        args.lr = 0.005
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None
        args.lr = 0.005
        args.n_clusters = 17
    args.input_dim = dataset.num_features
    print(args)
    pretrain(dataset)
