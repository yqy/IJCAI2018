# -*- coding: utf-8 -*-
import sys

import math
import random
import numpy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T
import torch.optim as optim

from conf import *

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

torch.manual_seed(args.random_seed)

class Network(nn.Module):

    def __init__(self, embedding_size, embedding_dimention, embedding_matrix, hidden_dimention, output_dimention):

        super(Network,self).__init__()

        self.embedding_layer = nn.Embedding(embedding_size,embedding_dimention)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.inpt_layer_zp_pre = nn.Linear(embedding_dimention,hidden_dimention)
        self.hidden_layer_zp_pre = nn.Linear(hidden_dimention,hidden_dimention)
        self.inpt_layer_zp_post = nn.Linear(embedding_dimention,hidden_dimention)
        self.hidden_layer_zp_post = nn.Linear(hidden_dimention,hidden_dimention)

        self.inpt_layer_np = nn.Linear(embedding_dimention,hidden_dimention)
        self.hidden_layer_np = nn.Linear(hidden_dimention,hidden_dimention)

        self.inpt_layer_nps = nn.Linear(hidden_dimention,hidden_dimention)
        self.hidden_layer_nps = nn.Linear(hidden_dimention,hidden_dimention)

        nh = hidden_dimention*2
        #nh = 2
        
        self.zp_pre_representation_layer = nn.Linear(hidden_dimention,nh)
        self.zp_post_representation_layer = nn.Linear(hidden_dimention,nh)
        self.np_representation_layer = nn.Linear(hidden_dimention,nh)
        self.nps_representation_layer = nn.Linear(hidden_dimention,nh)
        self.feature_representation_layer = nn.Linear(nnargs["feature_dimention"],nh)

        self.representation_hidden_layer = nn.Linear(hidden_dimention*2,hidden_dimention*2)
        self.output_layer = nn.Linear(hidden_dimention*2,output_dimention)
     
        self.hidden_size = hidden_dimention

        self.activate = nn.Tanh()
        self.softmax_layer = nn.Softmax()

    def forward_zp_pre(self, word_index, hiden_layer,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        word_embedding = self.embedding_layer(word_index)#.view(-1,word_embedding_rep_dimention)
        word_embedding = dropout_layer(word_embedding)
        this_hidden = self.inpt_layer_zp_pre(word_embedding) + self.hidden_layer_zp_pre(hiden_layer)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        return this_hidden

    def forward_zp_post(self, word_index, hiden_layer,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        word_embedding = self.embedding_layer(word_index)#.view(-1,word_embedding_rep_dimention)
        this_hidden = self.inpt_layer_zp_post(word_embedding) + self.hidden_layer_zp_post(hiden_layer)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        return this_hidden

    def forward_np(self, word_index, hiden_layer,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        word_embedding = self.embedding_layer(word_index)#.view(-1,word_embedding_rep_dimention)
        this_hidden = self.inpt_layer_np(word_embedding) + self.hidden_layer_np(hiden_layer)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        return this_hidden

    def forward_nps(self, inpt, hiden_layer,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        this_hidden = self.inpt_layer_nps(inpt) + self.hidden_layer_np(hiden_layer)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        return this_hidden

    def generate_score(self,zp_pre,zp_post,np,feature,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        x = self.zp_pre_representation_layer(zp_pre) + self.zp_post_representation_layer(zp_post) + self.np_representation_layer(np)\
            + self.feature_representation_layer(feature) 

        x = self.activate(x)
        x = dropout_layer(x)
        x = self.representation_hidden_layer(x)
        x = self.activate(x)
        x = dropout_layer(x)
        x = self.output_layer(x)
        #xs = F.sigmoid(x)
        xs = self.softmax_layer(x)

        return x,xs

    def generate_score_rl(self,zp_pre,zp_post,np,nps,feature,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        x = self.zp_pre_representation_layer(zp_pre) + self.zp_post_representation_layer(zp_post)\
            + self.np_representation_layer(np) + self.nps_representation_layer(nps)\
            + self.feature_representation_layer(feature) 

        x = self.activate(x)
        x = dropout_layer(x)
        x = self.representation_hidden_layer(x)
        x = self.activate(x)
        x = dropout_layer(x)
        x = self.output_layer(x)
        x = self.softmax_layer(x)

        return x


    def initHidden(self,batch=1):
        return autograd.Variable(torch.from_numpy(numpy.zeros((batch, self.hidden_size))).type(torch.cuda.FloatTensor))
            


def main():

    eb = np.array([[1.0,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6]])

    model = Network(6,3,eb,2,1).cuda()

    
    word_index = autograd.Variable(torch.cuda.LongTensor([[3,3,3],[2,2,2],[3,3,3],[1,1,1],[4,4,4]]))
    mask = autograd.Variable(torch.cuda.FloatTensor([[1,1,1],[0,1,1],[0,1,1],[0,0,1],[1,1,1]]))
    word_index = torch.transpose(word_index,0,1)
    mask = torch.transpose(mask,0,1)

    feature = [[1]*nnargs["feature_dimention"]]*11
    feature = autograd.Variable(torch.cuda.FloatTensor(feature))

    hidden_zp_pre = model.initHidden()
    for i in range(len(mask)):
        w = word_index[i]
        m = torch.transpose(mask[i:i+1],0,1) # to keep the dimention of mask as 3*1, not only 3
        hidden_zp_pre = model.forward_zp_pre(w,hidden_zp_pre)
        hidden_zp_pre = hidden_zp_pre*m

    hidden_zp_post = model.initHidden()
    for i in range(len(mask)):
        w = word_index[i]
        m = torch.transpose(mask[i:i+1],0,1) # to keep the dimention of mask as 3*1, not only 3
        hidden_zp_post = model.forward_zp_post(w,hidden_zp_post)
        hidden_zp_post = hidden_zp_post*m

    reindex = autograd.Variable(torch.cuda.LongTensor([0,0,0,1,1,1,1,1,3,4,2]))
   
    nps = [] 
    hidden_nps = model.initHidden()
    for re in reindex:
        #output = model.generate_score(hidden_zp_pre[re],hidden_zp_post[re],hidden_zp_pre[re],feature[re])
        output = model.generate_score_rl(hidden_zp_pre[re],hidden_zp_post[re],hidden_zp_pre[re],hidden_nps,feature[re])
        if output.data.cpu().numpy() >= 0.45:
            nps.append(hidden_zp_pre[re])
            hidden_nps = model.forward_nps(hidden_zp_pre[re],hidden_nps)
        print output

    '''
    target = torch.cuda.FloatTensor([[1],[1],[1],[0],[0],[0],[0],[1],[1],[1],[0]])

    optimizer = optim.SGD(model.parameters(), lr=1)
    optimizer.zero_grad()
    #loss = F.cross_entropy(output,autograd.Variable(target))
    print target
    loss = F.binary_cross_entropy(output,autograd.Variable(target))
    loss.backward()
    optimizer.step()

    hidden_zp_pre = model.initHidden()
    for i in range(len(mask)):
        w = word_index[i]
        m = torch.transpose(mask[i:i+1],0,1) # to keep the dimention of mask as 3*1, not only 3
        hidden_zp_pre = model.forward_zp_pre(w,hidden_zp_pre)
        hidden_zp_pre = hidden_zp_pre*m

    hidden_zp_post = model.initHidden()
    for i in range(len(mask)):
        w = word_index[i]
        m = torch.transpose(mask[i:i+1],0,1) # to keep the dimention of mask as 3*1, not only 3
        hidden_zp_post = model.forward_zp_post(w,hidden_zp_post)
        hidden_zp_post = hidden_zp_post*m

    reindex = autograd.Variable(torch.cuda.LongTensor([0,0,0,1,1,1,1,1,3,4,2]))

    output = model.generate_score(hidden_zp_pre[reindex],hidden_zp_post[reindex],hidden_zp_pre[reindex],feature)
    print output
    '''
if __name__ == "__main__":
    main()
