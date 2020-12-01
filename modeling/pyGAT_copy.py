import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.GATlayers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        # self.att_layer1 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        # self.att_layer2 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        # self.att_layer3 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        # self.attentions = [self.att_layer1,self.att_layer2,self.att_layer3]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self,w,h, x, adj =[]):
        x = F.dropout(x, self.dropout, training=self.training)
        cc =self.attentions[0](w,h,x, adj)
        b = cc[1]
        x = torch.cat([cc[0]], dim=1)
        # if not self.train():


        # else:
        #     b =None
        x = F.dropout(x, self.dropout, training=self.training)
        a = self.out_att(w,h,x, adj)[0]
        # b = self.out_att(w,h,x, adj)[1]
        x = F.elu(a)
        # return F.log_softmax(x, dim=1)
        return x,b

    # def forward(self,w,h, x, adj =[]):
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     cc =[att(w,h,x, adj)[0] for att in self.attentions]
    #     b = self.attentions[0](w, h, x, adj)[1]
    #     x = torch.cat(cc, dim=1)
    #     # if not self.train():
    #
    #
    #     # else:
    #     #     b =None
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     a = self.out_att(w,h,x, adj)[0]
    #     # b = self.out_att(w,h,x, adj)[1]
    #     x = F.elu(a)
    #     # return F.log_softmax(x, dim=1)
    #     return x,b

class Multi_GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        # self.att_layer1 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        # self.att_layer2 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        # self.att_layer3 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        # self.attentions = [self.att_layer1,self.att_layer2,self.att_layer3]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        return x

