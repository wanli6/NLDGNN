import numpy as np
import torch
import torch_geometric.nn as gnn
import torch.nn.functional as F
import math

from sklearn.metrics.pairwise import cosine_similarity
from torch import nn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)

        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class MLP(nn.Module):
    def __init__(self, nfeats, nhids, nclasses, dropout):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(nfeats, nhids)
        self.lin2 = nn.Linear(nhids, nclasses)
        self.act = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.lin1(x)
        x = self.act(x)
        x = self.dropout2(x)
        x = self.lin2(x)
        return x


class NLDGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclasses, k, nlayers, dropout, alpha, beta, gamma, eps, lamda, num_nodes, r,
                 init_v='mlp', weight=0.5):
        super(NLDGNN, self).__init__()
        self.lin1 = nn.Linear(nfeat, nhid)
        self.lin2 = nn.Linear(nhid, nclasses)
        self.lin3 = nn.Linear(nhid * 2 + nclasses, nclasses)
        self.vs = nn.ModuleList()
        for i in range(1):
            if init_v == 'gcn':
                self.vs.append(gnn.GCNConv(nhid, k, add_self_loops=True, cached=True))
            else:
                self.vs.append(nn.Linear(nhid, k))

        self.gate = nn.Linear(nclasses * 2, 1)
        self.att = nn.Linear(nclasses, 1, bias=False)
        self.left = nn.Linear(nclasses, nclasses, bias=False)
        self.lin_a = nn.Linear(num_nodes, nhid)
        self.lin_n = nn.Linear(nclasses, nclasses)
        self.W = nn.Parameter(torch.empty(nclasses, nclasses))
        nn.init.xavier_normal_(self.W)
        self.gamma = gamma
        self.hidden = nhid
        self.nlayers = nlayers
        self.eps = eps
        self.k = k
        self.init_v = init_v
        self.lamda = lamda
        # self.lamda = nn.Parameter(torch.tensor(lamda, dtype=torch.float32), requires_grad=True)
        # self.lamda1 = nn.Parameter(torch.Tensor([0., 0.]))
        self.alpha = alpha
        # self.alpha = nn.Parameter(torch.Tensor([0., 0., 0.]))
        self.beta = beta
        self.drp = dropout
        self.r = r
        self.confidence = nn.Parameter(torch.Tensor([0., 0., 0.]))
        self.matrix_drp = 1 - r
        self.b = nn.Parameter(torch.Tensor([0.]))
        self.weight1 = weight
        self.weight3 = None
        self.weight2 = nn.Parameter(torch.ones(num_nodes, 1))
        self.negative = nn.Parameter(torch.Tensor([0.1]))
        self.neigh_distribution = None

    def forward(self, x, adj, edge_index, edge_weight, connect, confidence):
        row, col = edge_index
        # 对结构信息进行MLP处理
        x_a = self.lin_a(connect)
        x = F.dropout(x, p=self.drp, training=self.training)
        # 对属性信息进行MLP处理
        x = self.lin1(x)
        # x_adj = torch.mm(adj, x)

        # 融合信息
        x = (1 - self.lamda) * x + self.lamda * x_a
        x = F.relu(x)
        x = F.dropout(x, p=self.drp, training=self.training)
        raw = x

        x = self.lin2(x)

        # else:
        #
        #     raw = x
        #     x = torch.cat((x, x_a), dim=1)
        #     x = torch.cat((x, self.neigh_distribution), dim=1)
        #     x = F.relu(x)
        #     x = F.dropout(x, p=self.drp, training=self.training)
        #
        #     x = self.lin3(x)

        if self.neigh_distribution is None:
            # self.neigh_distribution = self.get_neighbor_distribution(x, edge_index)
            self.neigh_distribution = self.get_neighbor_distribution2(x, adj)

        x_n = self.lin_n(self.neigh_distribution)
        x_n = F.relu(x_n)
        x_n = F.dropout(x_n, p=self.drp, training=self.training)

        # x_n = self.neigh_distribution

        # self.weight3 = self.cal_nei_dis(x.size(1), adj, x.size(0), x).to(x.device)
        x = (1 - self.weight1) * x + self.weight1 * x_n
        init = x

        # if not self.training:
        #     torch.save(x, 'save/squirrel_h.pt')

        confi_att = F.relu(self.left(x)[row] + self.left(x)[col])
        confi_att = F.sigmoid(self.att(confi_att).squeeze(-1))
        w = F.softmax(self.confidence, dim=0)
        confidence = confi_att * w[0] + confidence[0] * w[1] + confidence[1] * w[2]


        for i in range(1):
            if self.init_v == "gcn":
                V = self.vs[i](raw, edge_index)
            else:
                V = self.vs[i](raw)
            U = (torch.sparse.mm(adj, V))
            inver = torch.inverse(torch.mm(V.T, V))
            U = torch.mm(U, inver)

            if self.training:
                drp = torch.bernoulli(torch.Tensor([self.matrix_drp] * x.size(1))).to(x.device) / self.matrix_drp
                drp_x = x * drp.view(1, -1)
                drp = torch.bernoulli(torch.Tensor([self.matrix_drp] * self.k)).to(x.device) / self.matrix_drp
            else:
                drp_x = x
                drp = 1

            for j in range(self.gamma):
                if j % 2 == 0:
                    U = self.update_u(U, V, x, edge_weight, edge_index, drp_x, drp, confidence)
                else:
                    V = self.update_v(U, V, x, edge_weight, edge_index, drp_x, drp, confidence)
            xt_v = torch.mm(x.T, V)
            prop_x = torch.mm(U, xt_v.T)

            # z = self.update_z(z, adj, x)
            # z = self.update_z1(z, x, adj)
            # z = self.update_z2(adj)

            # z = torch.mm(U, V.T)
            # # prop_x = torch.mm(z, x)
            # # if self.training is False:
            # #     print("Save Z:")
            # #     torch.save(z, 'visual/z.pt')
            #
            # z_positive = torch.clamp(z, min=0)
            # z_negative = torch.clamp(z, max=0)
            #
            # prop_x_positive = torch.mm(z_positive, x)
            # prop_x_negative = torch.mm(z_negative, x)
            # prop_x = prop_x_positive + self.negative * prop_x_negative

            # prop_x = torch.mm(adj, x)
            # prop_x = torch.mm(z, x)
            # prop_x = (1 - self.weight1) * prop_x + self.weight1 * x_n
            # prop_x = prop_x + self.weight2 * x_n
            x = (1 - self.alpha) * prop_x + self.alpha * init
            x = torch.mm(x, self.W)
            # self.neigh_distribution = self.get_neighbor_distribution(x, edge_index)
            self.neigh_distribution = self.get_neighbor_distribution2(x, adj)
        return x

    def update_z(self, z, adj, x):

        return z

    def get_neighbor_distribution(self, x, edge_index):
        x = x.softmax(dim=1)
        y_hat = torch.argmax(x, dim=1)
        neigh_distribution = []
        for node_idx in range(x.size(0)):
            neighbors = edge_index[1, edge_index[0] == node_idx].tolist()
            neighbors_label = y_hat[neighbors]
            distri = torch.bincount(neighbors_label, minlength=x.size(1)).float()
            distri = distri / distri.sum() if distri.sum() > 0 else distri
            neigh_distribution.append(distri)
        neigh_distribution = torch.stack(neigh_distribution)
        return neigh_distribution

    def get_neighbor_distribution2(self, x, adj):
        x = x.softmax(dim=1)
        y_hat = torch.argmax(x, dim=1)
        one_hot = torch.eye(x.size(1)).to(y_hat.device)
        one_hot = one_hot[y_hat]
        nei_dis = torch.sparse.mm(adj, one_hot)
        # nei_dis = torch.matmul(adj, nei_dis)
        # nei_dis = nei_dis + one_hot
        nei_dis_sum = nei_dis.sum(dim=1, keepdim=True)
        nei_dis = torch.where(nei_dis_sum != 0, nei_dis / nei_dis_sum, torch.zeros_like(nei_dis))
        # nei_dis = nei_dis + one_hot
        return nei_dis

    # def update_z1(self, z, x, adj):
    #     x = x.softmax(dim=1)
    #     y_hat = torch.argmax(x, dim=1)
    #     one_hot = torch.eye(x.size(1)).to(y_hat.device)
    #     one_hot = one_hot[y_hat]
    #     nei_dis = torch.matmul(adj, one_hot)
    #     # nei_dis = nei_dis + one_hot
    #     nei_dis_sum = nei_dis.sum(dim=1, keepdim=True)
    #     nei_dis = torch.where(nei_dis_sum != 0, nei_dis / nei_dis_sum, torch.zeros_like(nei_dis))
    #
    #     similarity_matrix = torch.nn.functional.cosine_similarity(nei_dis.unsqueeze(1),
    #                                                               nei_dis.unsqueeze(0), dim=-1)
    #
    #     # z = self.weight * z + (1-self.weight) * similarity_matrix
    #     # w = torch.softmax(self.weight, dim=0)
    #     # z = w[0] * z + w[1] * similarity_matrix
    #     z = (1-self.weight) * z + self.weight * similarity_matrix
    #     return z

    def update_u(self, U, V, x, edge_weight, edge_index, drp_x, drp, confidence):
        row, col = edge_index
        if self.training:
            r_v = drp.view(1, -1) * V
        else:
            r_v = V

        score = (edge_weight - ((U[row] * V[col]).sum(dim=1)))
        S = score ** 2
        sorted_s = torch.sort(S)[0]
        edge_size = edge_weight.size(0)
        first_quartile = sorted_s[edge_size // 4]
        third_quartile = sorted_s[3 * edge_size // 4]
        threshold = (third_quartile + 1.5 * (third_quartile - first_quartile))
        threshold = threshold * confidence
        confidence = torch.where(S < threshold, torch.Tensor([1.]).to(x.device), torch.Tensor([0.]).to(x.device))
        SS = (score + self.b) * confidence
        # mask = S < threshold
        # SS[mask] = threshold[mask]
        sparse_adj = torch.sparse_coo_tensor(edge_index, SS, [x.size(0), x.size(0)])
        # sparse_adj = torch.sparse_coo_tensor(edge_index, (score + self.b) , [x.size(0), x.size(0)])
        temp = torch.sparse.mm(sparse_adj, V) + torch.mm(U, torch.mm(V.T, V))
        U = (temp + self.eps * torch.mm(x, torch.mm(drp_x.T, V)))
        # U = temp

        inver = torch.inverse(
            (torch.mm(V.T, V) + self.eps * torch.mm(r_v.T, V)
             + self.beta * torch.eye(self.k, device=x.device)
             )
        )
        U = torch.mm(U, inver)
        return U

    def update_v(self, U, V, x, edge_weight, edge_index, drp_x, drp, confidence):
        if self.training:
            r_u = drp.view(1, -1) * U
        else:
            r_u = U
        inver = torch.inverse(
            torch.mm(U.T, U) + self.eps * torch.mm(r_u.T, U) + self.beta * torch.eye(U.size(1), device=x.device))
        row, col = edge_index
        score = (edge_weight - (V[row] * U[col]).sum(dim=1))
        S = score ** 2
        sorted_s = torch.sort(S)[0]
        edge_size = edge_weight.size(0)
        first_quartile = sorted_s[edge_size // 4]
        third_quartile = sorted_s[3 * edge_size // 4]
        threshold = (third_quartile + 1.5 * (third_quartile - first_quartile))
        threshold = threshold * confidence
        confidence = torch.where(S < threshold, torch.Tensor([1.]).to(x.device), torch.Tensor([0.]).to(x.device))
        SS = (score + self.b) * confidence
        # mask = S < threshold
        # SS[mask] = threshold[mask]
        sparse_adj = torch.sparse_coo_tensor(edge_index, SS, [x.size(0), x.size(0)])
        temp = torch.sparse.mm(torch.transpose(sparse_adj, 0, 1), U) + \
               torch.mm(V, torch.mm(U.T, U))
        V = temp + self.eps * torch.mm(x, torch.mm(drp_x.T, U))
        # V = temp

        return torch.mm(V, inver)

    def cal_nei_dis(self, c, adj, num_nodes, x):
        # edge_index, _ = remove_self_loops(data.edge_index)
        nei_distri = self.get_neighbor_distribution2(x, adj)
        intra_dis = torch.zeros((c, 1))
        inter_dis = torch.zeros((c, c))
        x = x.softmax(dim=1)
        y_hat = torch.argmax(x, dim=1)
        for label in range(c):
            nei_distri_label = nei_distri[y_hat == label]
            inter_dis[label] = torch.mean(nei_distri_label, dim=0)
            # intra_nd = torch.mean(torch.sum(nei_distri_label ** 2, dim=1) - 1 / c)
            # intra_dis.append(intra_nd * (c /(c-1)))
            matrix = torch.nn.functional.cosine_similarity(nei_distri_label.unsqueeze(1), nei_distri_label.unsqueeze(0),
                                                           dim=-1)
            matrix = (matrix + 1) / 2
            intra_dis[label] = torch.mean(matrix.view(-1)) * (nei_distri_label.size(0) / num_nodes)

        intra = torch.sum(intra_dis, dim=0)
        # inter_dis_sum = torch.sum(inter_dis, dim=0, keepdim=True)
        # inter_dis = torch.where(inter_dis_sum > 0, inter_dis / inter_dis_sum, torch.zeros_like(inter_dis))
        # # print(inter_dis)
        # inter = torch.mean(torch.sum(inter_dis**2, dim=1)-(1 / c))*(c/(c-1))
        print(inter_dis)
        singular_values = torch.linalg.svdvals(inter_dis)
        normalize_singular = singular_values / torch.sum(singular_values)
        be_sum = normalize_singular * torch.log(normalize_singular)
        to_sum = torch.nan_to_num(be_sum, 0)
        inter = -torch.sum(to_sum) / torch.log(torch.tensor(c))

        nd = (intra + inter) / 2
        return nd
