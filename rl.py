#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy
import random
import copy
import cPickle
sys.setrecursionlimit(1000000)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T
from torch.optim import lr_scheduler

from conf import *
import utils
from data_generater import *
from net import *
import performance
random.seed(args.random_seed)
numpy.random.seed(args.random_seed)

print >> sys.stderr, "PID", os.getpid()

torch.cuda.set_device(args.gpu)

def net_copy(net,copy_from_net):
    mcp = list(net.parameters())
    mp = list(copy_from_net.parameters())
    n = len(mcp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:]

def main():
    #train_generater = DataGnerater("train",0)
    #train_generater = DataGnerater("test",0)
    train_generater = DataGnerater("train",nnargs["batch_size"])
    #train_generater = DataGnerater("test",nnargs["batch_size"])
    test_generater = DataGnerater("test",256)

    embedding_matrix = numpy.load(args.data + "embedding.npy")
    print "Building torch model"

    model = Network(nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix,nnargs["hidden_dimention"],2).cuda()
    best_model_ = torch.load("./model/model.pretrain.best")
    net_copy(model,best_model_)

    best_model = model

    performance.get_performance(test_generater,model) 

    this_lr = nnargs["lr"]
    #optimizer = optim.Adagrad(model.parameters(),lr=this_lr)
    best_result = {}
    best_result["hits"] = 0

    for echo in range(nnargs["epoch"]):
        cost = 0.0
        baseline = []
        print >> sys.stderr, "Begin epoch",echo
        #optimizer = optim.Adadelta(model.parameters(),lr=this_lr)
        #optimizer = optim.Adam(model.parameters(),lr=this_lr)
        #optimizer = optim.RMSprop(model.parameters(),lr=this_lr)

        optimizer = optim.Adagrad(model.parameters(),lr=this_lr) #best
        this_lr = this_lr*0.9

        for data in train_generater.generate_data(shuffle=True):
            #zp
            zp_reindex = autograd.Variable(torch.from_numpy(data["zp_reindex"]).type(torch.cuda.LongTensor))
            zp_pre = autograd.Variable(torch.from_numpy(data["zp_pre"]).type(torch.cuda.LongTensor))
            zp_pre_mask = autograd.Variable(torch.from_numpy(data["zp_pre_mask"]).type(torch.cuda.FloatTensor))
            zp_post = autograd.Variable(torch.from_numpy(data["zp_post"]).type(torch.cuda.LongTensor))
            zp_post_mask = autograd.Variable(torch.from_numpy(data["zp_post_mask"]).type(torch.cuda.FloatTensor))
            
            #np
            candi_reindex = autograd.Variable(torch.from_numpy(data["candi_reindex"]).type(torch.cuda.LongTensor))
            candi = autograd.Variable(torch.from_numpy(data["candi"]).type(torch.cuda.LongTensor))
            candi_mask = autograd.Variable(torch.from_numpy(data["candi_mask"]).type(torch.cuda.FloatTensor))
            
            feature = autograd.Variable(torch.from_numpy(data["fl"]).type(torch.cuda.FloatTensor))
    
            zp_pre = torch.transpose(zp_pre,0,1)
            mask_zp_pre = torch.transpose(zp_pre_mask,0,1)
            hidden_zp_pre = model.initHidden()
            for i in range(len(mask_zp_pre)):
                hidden_zp_pre = model.forward_zp_pre(zp_pre[i],hidden_zp_pre,dropout=nnargs["dropout"])*torch.transpose(mask_zp_pre[i:i+1],0,1)
            zp_pre_representation = hidden_zp_pre[zp_reindex]
    
            zp_post = torch.transpose(zp_post,0,1)
            mask_zp_post = torch.transpose(zp_post_mask,0,1)
            hidden_zp_post = model.initHidden()
            for i in range(len(mask_zp_post)):
                hidden_zp_post = model.forward_zp_post(zp_post[i],hidden_zp_post,dropout=nnargs["dropout"])*torch.transpose(mask_zp_post[i:i+1],0,1)
            zp_post_representation = hidden_zp_post[zp_reindex]
    
            candi = torch.transpose(candi,0,1)
            mask_candi = torch.transpose(candi_mask,0,1)
            hidden_candi = model.initHidden()
            for i in range(len(mask_candi)):
                hidden_candi = model.forward_np(candi[i],hidden_candi,dropout=nnargs["dropout"])*torch.transpose(mask_candi[i:i+1],0,1)
            candi_representation = hidden_candi[candi_reindex]

            output,output_softmax = model.generate_score(zp_pre_representation,zp_post_representation,candi_representation,feature,dropout=nnargs["dropout"])

            noneed = output_softmax[:,0].data.cpu().numpy()
            need = output_softmax[:,1].data.cpu().numpy()
            trick = need-noneed
            path = numpy.clip(numpy.floor(trick), -1, 0).astype(int)+1

            #thres = output_softmax[:,1].data.cpu().numpy()
            #ran = numpy.random.rand(len(thres))
            #path = numpy.clip(numpy.floor(ran / thres), 1, 0).astype(int)

            gold = data["result"]
            if float(sum(gold)) == 0 or sum(gold*path) == 0 or sum(path) == 0:
                continue
            precision = float(sum(gold*path))/float(sum(path))
            recall = float(sum(gold*path))/float(sum(gold))
            f = 2.0/(1.0/precision+1.0/recall) if (recall > 0 and precision > 0) else 0.0
            if f == 0:
                continue

            '''
            hit = 0
            for i in range(len(path)):
                if path[i] == gold[i]:
                    hit += 1
            f = float(hit)/float(len(gold))
            if f == 0:
                continue
            '''

            #reward = -1.0*f
            #base = sum(baseline)/float(len(baseline)) if len(baseline) > 0 else 0.0
            #reward = -1.0*(f-base)
            #baseline.append(f)
            #if len(baseline) > 64:
            #    baseline = baseline[1:]
 
            rewards = numpy.full((len(path),2),f)
            path_list = path.tolist()

            for i in range(len(path_list)):
                new_path = copy.deepcopy(path_list)
                new_path[i] = 1-new_path[i]

                if float(sum(gold)) == 0 or sum(gold*new_path) == 0 or sum(new_path) == 0:
                    new_f = 0.0
                else:
                    new_precision = float(sum(gold*new_path))/float(sum(new_path))
                    new_recall = float(sum(gold*new_path))/float(sum(gold))
                    new_f = 2.0/(1.0/new_precision+1.0/new_recall) if (new_recall > 0 and new_precision > 0) else 0.0
                rewards[i][new_path[i]] = new_f
            
            #maxs = rewards.max(axis=1)[:,numpy.newaxis]
            maxs = rewards.min(axis=1)[:,numpy.newaxis]
            rewards = rewards - maxs
            rewards *= -10.0

            rewards = autograd.Variable(torch.from_numpy(rewards).type(torch.cuda.FloatTensor))
            
            optimizer.zero_grad()
            #output_softmax = torch.log(output_softmax)
            loss = torch.sum(output_softmax*rewards)
            #loss = torch.mean(output_softmax*rewards)
            cost += loss.data[0]
            loss.backward()
            optimizer.step()
        print >> sys.stderr, "End epoch",echo,"Cost:", cost

        print "Result for epoch",echo 
        result = performance.get_performance(test_generater,model) 
        if result["hits"] >= best_result["hits"]:
            best_result = result
            best_result["epoch"] = echo
            best_model = model
            torch.save(best_model, "./model/model.best")

    print "Best Result on epoch", best_result["epoch"]
    print "Hits",best_result["hits"]
    print "R",best_result["r"],"P",best_result["p"],"F",best_result["f"]
if __name__ == "__main__":
    main()
