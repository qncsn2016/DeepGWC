from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import torch

class noflayer(nn.Module):
    def __init__(self, bala3, nnode, in_features, out_features, residual=False, variant=False):
        super(noflayer, self).__init__()
        self.bala3=bala3
        self.variant = variant
        self.nnode = nnode
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        # self.f = Parameter(torch.ones(self.nnode))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h0, lamda, alpha, l):
        beta = math.log(lamda/l+1)
        hi=torch.mm(self.bala3, input)
        # hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = beta*torch.mm(support, self.weight)+(1-beta)*r
        if self.residual:
            output = output+input
        return output

class combinelayer(nn.Module):
    def __init__(self, nnode, in_features, out_features, residual=False, variant=False):
        super(combinelayer, self).__init__()
        self.variant = variant
        self.nnode = nnode
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.f = Parameter(torch.ones(self.nnode))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, support0, support1, adj, gamma, h0, lamda, alpha, l):
        beta = math.log(lamda/l+1)

        bala1=torch.spmm(support0,torch.diag(self.f))
        bala2=torch.mm(bala1,support1)

        bala3=gamma*bala2+(1-gamma)*adj

        hi=torch.mm(bala3, input)
        # hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = beta*torch.mm(support, self.weight)+(1-beta)*r
        if self.residual:
            output = output+input
        return output

class WaveletConvolution(nn.Module):
    def __init__(self, nnode, in_features, out_features, residual=False, variant=False):
        super(WaveletConvolution, self).__init__()
        self.variant = variant
        self.nnode = nnode
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.f = Parameter(torch.ones(self.nnode))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, support0, support1, h0, lamda, alpha, l):
        beta = math.log(lamda/l+1)

        bala1=torch.spmm(support0,torch.diag(self.f))
        bala2=torch.mm(bala1,support1)
        hi=torch.mm(bala2, input)
        # hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = beta*torch.mm(support, self.weight)+(1-beta)*r
        if self.residual:
            output = output+input
        return output

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output
