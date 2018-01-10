#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy
import random
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
random.seed(args.random_seed)

print >> sys.stderr, "PID", os.getpid()

torch.cuda.set_device(args.gpu)

def net_copy(net,copy_from_net):
    mcp = list(net.parameters())
    mp = list(copy_from_net.parameters())
    n = len(mcp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:]

def get_predict(data,t):
    predict = []
    for result,output in data:
        max_index = -1
        for i in range(len(output)):
            if output[i][1] > t:
                max_index = i
        predict.append(result[max_index])
    return predict

def get_predict_max(data):
    predict = []
    for result,output in data:
        max_index = -1
        max_pro = 0.0
        for i in range(len(output)):
            if output[i][1] > max_pro:
                max_index = i
                max_pro = output[i][1]
        predict.append(result[max_index])
    return predict
 
 
def get_evaluate(data):
    res_t = [0.3,0.35,0.4,0.45,0.5,0.55,0.6]
    best_result = {}
    best_result["hits"] = 0

    # best first
    predict = get_predict_max(data)
    result = evaluate(predict)
    if result["hits"] > best_result["hits"]:
        best_result = result
        best_result["t"] = -1

    # nearest first
    for t in res_t:
        predict = get_predict(data,t)
        result = evaluate(predict)
        if result["hits"] > best_result["hits"]:
            best_result = result
            best_result["t"] = t

    print >> sys.stderr, "Hits",best_result["hits"]
    print "Hits",best_result["hits"]
    print "Best thresh",best_result["t"]
    print "R",best_result["r"],"P",best_result["p"],"F",best_result["f"]
    print
    return best_result

def evaluate(predict):
    result = {}
    result["hits"] = sum(predict)
    p = sum(predict)/float(len(predict))
    r = sum(predict)/1713.0
    f = 0.0 if (p == 0 or r == 0) else (2.0/(1.0/p+1.0/r))
    result["r"] = r
    result["p"] = p
    result["f"] = f
    return result

MAX = 2

def main():
    train_generater = DataGnerater("train",nnargs["batch_size"])
    test_generater = DataGnerater("test",256)

    embedding_matrix = numpy.load(args.data + "embedding.npy")
    print "Building torch model"

    model = Network(nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix,nnargs["hidden_dimention"],2).cuda()

    this_lr = nnargs["lr"]
    optimizer = optim.Adagrad(model.parameters(), lr=this_lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    best_result = {}
    best_result["hits"] = 0
    best_model = None
     
    for echo in range(nnargs["epoch"]):
        cost = 0.0
        scheduler.step()
        print >> sys.stderr, "Begin epoch",echo
    
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
    
            assert len(feature) == len(candi_representation)
            assert len(zp_post_representation) == len(candi_representation)
            output,output_softmax = model.generate_score(zp_pre_representation,zp_post_representation,candi_representation,feature,dropout=nnargs["dropout"])
    
            #target = autograd.Variable(torch.from_numpy(data["target"]).type(torch.cuda.FloatTensor)) 
            #target = autograd.Variable(torch.from_numpy(data["target"]).type(torch.cuda.LongTensor)) 
            target = autograd.Variable(torch.from_numpy(data["result"]).type(torch.cuda.LongTensor)) 
            
            optimizer.zero_grad()
            #loss = F.binary_cross_entropy(output,target)
            loss = F.cross_entropy(output,target)
            cost += loss.data[0]
            loss.backward()
            optimizer.step()
        print >> sys.stderr, "End epoch",echo,"Cost:", cost
    
        predict = []
        for data in test_generater.generate_data():
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
                hidden_zp_pre = model.forward_zp_pre(zp_pre[i],hidden_zp_pre)*torch.transpose(mask_zp_pre[i:i+1],0,1)
            zp_pre_representation = hidden_zp_pre[zp_reindex]
    
            zp_post = torch.transpose(zp_post,0,1)
            mask_zp_post = torch.transpose(zp_post_mask,0,1)
            hidden_zp_post = model.initHidden()
            for i in range(len(mask_zp_post)):
                hidden_zp_post = model.forward_zp_post(zp_post[i],hidden_zp_post)*torch.transpose(mask_zp_post[i:i+1],0,1)
            zp_post_representation = hidden_zp_post[zp_reindex]
    
            candi = torch.transpose(candi,0,1)
            mask_candi = torch.transpose(candi_mask,0,1)
            hidden_candi = model.initHidden()
            for i in range(len(mask_candi)):
                hidden_candi = model.forward_np(candi[i],hidden_candi)*torch.transpose(mask_candi[i:i+1],0,1)
            candi_representation = hidden_candi[candi_reindex]
    
            output,output_softmax = model.generate_score(zp_pre_representation,zp_post_representation,candi_representation,feature)
            output_softmax = output_softmax.data.cpu().numpy()
            for s,e in data["start2end"]:
                if s == e:
                    continue
                predict.append((data["result"][s:e],output_softmax[s:e]))

        print "Result for epoch",echo 
        result = get_evaluate(predict)
        if result["hits"] > best_result["hits"]:
            best_result = result
            best_result["epoch"] = echo 
            best_model = model 
            torch.save(best_model, "./model/model.best")
        sys.stdout.flush()

    print "Best Result on epoch", best_result["epoch"]
    print "Hits",best_result["hits"]
    print "R",best_result["r"],"P",best_result["p"],"F",best_result["f"]
 
if __name__ == "__main__":
    main()
