import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.GATlayers import GraphAttentionLayer, SpGraphAttentionLayer

def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))


        self.att1_conv1 = init_(nn.Conv1d(in_channels=nfeat, out_channels=nhid, kernel_size=1))
        self.att2_conv1 = init_(nn.Conv1d(in_channels=nfeat, out_channels=nhid, kernel_size=1))
        self.att3_conv1 = init_(nn.Conv1d(in_channels=nfeat, out_channels=nhid, kernel_size=1))
        # self.attentions_1 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        # self.attentions_2 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        # self.attentions_3 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        # self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
        #                    range(nheads)]
        # self.attentions = [self.attentions_1,self.attentions_2,self.attentions_3]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        self.n_class = nclass
        self.out_att_conv1 = init_(nn.Conv1d(in_channels=nhid * nheads, out_channels=nclass, kernel_size=1))

    def forward(self, x, adj =[]):
        attention_heads = []
        h_size = x.size()[-2]
        w_size = x.size()[-1]
        x = x.view(x.size()[1], w_size * h_size)
        x = x.permute(1, 0)
        x = nn.functional.dropout(x, self.dropout, training=self.training)

        h = self.att1_conv1(x.unsqueeze(-1)).squeeze(-1)
        h_1 = nn.functional.normalize(h, dim=1)
        # h_1 =torch.softmax(h,dim=1)
        h_t = torch.t(h_1)
        e = torch.matmul(h_1, h_t)
        attention = nn.functional.normalize(e,dim=0)
        # attention = torch.softmax(e, dim=0)
        h = torch.matmul(attention, h_1)
        h=nn.functional.elu(h)

        attention_heads.append(h)
        h = self.att2_conv1(x.unsqueeze(-1)).squeeze(-1)
        h_1 = nn.functional.normalize(h, dim=1)
        # h_1 =torch.softmax(h,dim=1)
        h_t = torch.t(h_1)
        e = torch.matmul(h_1, h_t)
        attention = nn.functional.normalize(e,dim=0)
        # attention = torch.softmax(e, dim=0)
        h = torch.matmul(attention, h_1)
        h = nn.functional.elu(h)
        attention_heads.append(h)
        h = self.att3_conv1(x.unsqueeze(-1)).squeeze(-1)
        h_1 = nn.functional.normalize(h, dim=1)
        # h_1 =torch.softmax(h,dim=1)
        h_t = torch.t(h_1)
        e = torch.matmul(h_1, h_t)
        attention = nn.functional.normalize(e,dim=0)
        # attention = torch.softmax(e, dim=0)
        h = torch.matmul(attention, h_1)
        h = nn.functional.elu(h)
        attention_heads.append(h)
        # attention_heads =[att(w,h,x, adj) for att in self.attentions]
        x = torch.cat(attention_heads, dim=1)
        x = nn.functional.dropout(x, self.dropout, training=self.training)
        h = self.out_att_conv1(x.unsqueeze(-1)).squeeze(-1)
        h_1 = nn.functional.normalize(h, dim=1)
        # h_1 =torch.softmax(h,dim=1)
        h_t = torch.t(h_1)
        e = torch.matmul(h_1, h_t)
        attention = nn.functional.normalize(e,dim=0)
        # attention = torch.softmax(e, dim=0)


        h = torch.matmul(attention, h_1)
        h = h.unsqueeze(-1)
        h_1 = h.permute(2,1,0)
        # x = nn.functional.elu(a)
        # print(h,w)
        h_1 = h_1.view(1, self.n_class, h_size, w_size)
        # h =nn.functional.log_softmax(h_1,dim =1)
        # if not self.training:
        #     heatmap = {}
        #     for i in range(h_size):
        #         for j in range(w_size):
        #             heatmap[(i, j)] = []
        #             kk = i * w_size + j
        #             resize_a = attention[:, kk]
        #             # print(resize_a)
        #             top5_list = resize_a.topk(10).indices.data
        #             # print(top5_list)
        #             for wh in top5_list:
        #                 w_index = wh.cpu().data % w_size
        #                 h_index = wh.cpu().data / w_size
        #                 heatmap[(i, j)].append((int(h_index.data),int(w_index.data)))
        #     return h_1,heatmap
        return h_1