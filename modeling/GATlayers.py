import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        # if not concat:
        # init_ = lambda m: init(m, nn.init.orthogonal_,
        #                        lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1)


        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        # # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.conv1 = nn.Conv1d(in_channels=in_features,out_channels=out_features,kernel_size=1)
        # init_ = lambda m: init(m, nn.init.orthogonal_,
        #                        lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        # self.a_conv2 = init_(nn.Conv1d(in_channels=out_features,out_channels=1,kernel_size=1))
        # self.a_conv3 = init_(nn.Conv1d(in_channels=out_features, out_channels=1, kernel_size=1))
        # # self.atten_conv1 = init_(nn.Conv2d(in_channels=512, out_channels=attention_hidden_size, kernel_size=1))
        # # self.atten_conv2 = init_(nn.Conv2d(in_channels=attention_hidden_size, out_channels=1, kernel_size=1))
        # self.a = nn.Parameter(torch.zeros(size=(2, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        #
        #
        # self.leakyrelu = nn.LeakyReLU(self.alpha)

    # def forward(self, w_size, h_size, h, adj=[]):
    #     # h = torch.mm(input, self.W)
    #     h = torch.unsqueeze(h, 2)
    #     h = self.conv1(h)
    #     # h = torch.squeeze(self.conv1(h))
    #     # h_t = torch.t(h)
    #     # e = torch.matmul(h, h_t)
    #     att_h1 = self.a_conv1(h)
    #     att_h2 = self.a_conv2(h)
    #
    #     h = torch.squeeze(h, 2)
    #     att_h2 = torch.squeeze(att_h2, 2)
    #     att_h1 = torch.squeeze(att_h1, 2)
    #     N = h.size()[0]
    #
    #     a_input = torch.cat([att_h1.repeat(1, N).view(N * N, -1), att_h2.repeat(N, 1)], dim=1).view(N, N, 2)
    #     e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
    #
    #     # zero_vec = -9e15*torch.ones_like(e)
    #     attention = e
    #     # # attention = torch.where(adj > 0, e, zero_vec)
    #     attention = F.softmax(attention, dim=1)
    #     heatmap = {}
    #     for i in range(w_size):
    #         for j in range(h_size):
    #             heatmap[(i, j)] = []
    #             a = attention[i*h_size+j]
    #             # print(a)
    #             top5_list = list(a.topk(5).indices.data)
    #             print(top5_list)
    #             for wh in top5_list:
    #                 w_index = wh%h_size
    #                 h_index = wh%w_size
    #                 heatmap[(i,j)].append((int(w_index.data),int(h_index.data)))
    #
    #     attention = F.dropout(attention, self.dropout, training=self.training)
    #     h_prime = torch.matmul(attention, h)
    #
    #     if self.concat:
    #         return F.elu(h_prime), heatmap
    #     else:
    #         return h_prime, heatmap

    def forward(self, w_size,h_size, h, adj=[]):
        # if not self.concat:
        h = self.conv1(h.unsqueeze(-1)).squeeze(-1)
        h_1 = nn.functional.normalize(h,dim=1)
        # h_1 =torch.softmax(h,dim=1)
        h_t = torch.t(h_1)
        e = torch.matmul(h_1, h_t)
        # attention = nn.functional.normalize(e,dim=0)
        attention = torch.softmax(e,dim = 0)
        att =torch.matmul(attention, h_1)
        h =att +h_1
        heatmap = {}
        for i in range(h_size):
            for j in range(w_size):
                heatmap[(i, j)] = []
                kk = i*w_size+j
                resize_a = attention[:,kk]
                # print(resize_a)
                top5_list = resize_a.topk(5).indices.data
                # print(top5_list)
                for wh in top5_list:
                    w_index = wh.cpu().data%w_size
                    h_index = wh.cpu().data/w_size
                    heatmap[(i,j)].append((int(w_index.data),int(h_index.data)))
        # attention = F.dropout(attention, self.dropout, training=self.training)
        # h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h),heatmap
        else:

            return h.unsqueeze(-1),heatmap

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1)).cuda())
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
