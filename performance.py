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
import torch.optim as optim

from conf import *
import utils
from data_generater import *
from net import *
random.seed(args.random_seed)

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
    best_result = {}
    best_result["hits"] = 0

    # best first
    predict = get_predict_max(data)
    result = evaluate(predict)
    best_result = result
    best_result["t"] = -1

    print >> sys.stderr, "Hits",best_result["hits"]
    print "Hits",best_result["hits"]
    print "Best thresh",best_result["t"]
    print "R",best_result["r"],"P",best_result["p"],"F",best_result["f"]

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

def get_performance(test_generater,model):
    predict = []
    HITS = 0
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
        predict.append((data["result"],output_softmax))

    get_evaluate(predict)
    #print "Use RL Method Hits,", HITS
    #print >> sys.stderr, "Use RL Method Hits,", HITS
    print
    sys.stdout.flush()
if __name__ == "__main__":
    main()
