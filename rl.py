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
    test_generater = DataGnerater("test",0)

    embedding_matrix = numpy.load(args.data + "embedding.npy")
    print "Building torch model"

    model = Network(nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix,nnargs["hidden_dimention"],2).cuda()
    best_model = torch.load("./model/model.best")
    net_copy(model,best_model)

    performance.get_performance(test_generater,model) 
    #performance.get_performance(train_generater,model) 

    this_lr = nnargs["lr"]
    #optimizer = optim.Adagrad(model.parameters(),lr=this_lr)
    optimizer = optim.Adadelta(model.parameters(),lr=this_lr)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99) # go 100 times and reduce lr=lr*0.99

    for echo in range(nnargs["epoch"]):
        cost = 0.0
        baseline = []
        print >> sys.stderr, "Begin epoch",echo
        for data in train_generater.generate_data(shuffle=True):
            #scheduler.step()
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

            path = []
            nps = []
            output,output_softmax = model.generate_score(zp_pre_representation,zp_post_representation,candi_representation,feature,dropout=nnargs["dropout"])
            for i in range(len(output_softmax)):
                prob = output_softmax[i]
                action_probs = prob.data.cpu().numpy()
                #if action_probs[1] > action_probs[0]:
                if random.random() <= action_probs[1]:
                    nps.append(i)
                    path.append(1)
                else:
                    path.append(0)

            '''
            nps = []
            hidden_nps = model.initHidden()
            path = []
            for (zpi,i) in zip(zp_reindex,candi_reindex):
                output_softmax = model.generate_score_rl(zp_pre_representation[zpi],zp_post_representation[zpi],candi_representation[i],hidden_nps,feature[i],dropout=nnargs["dropout"])
                action_probs = output_softmax.data.cpu().numpy()[0][1]
                need_index = i.data.cpu().numpy()[0]
                if random.random() <= action_probs: # regard it as an output
                    nps.append(need_index)
                    hidden_nps = model.forward_nps(candi_representation[i],hidden_nps)
                    path.append(1)
                else:
                    path.append(0)
            '''

            gold = data["result"]
            if float(sum(gold)) == 0 or len(gold[nps]) == 0:
                continue
            precision = float(sum(gold[nps]))/float(len(gold[nps]))
            recall = float(sum(gold[nps]))/float(sum(gold))
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
 
            rewards = []
            for i in range(len(output_softmax)):
                new_nps = copy.deepcopy(nps)
                pa = path[i]
                if pa == 1:
                    new_nps.remove(i)
                else:
                    new_nps.append(i)

                if float(sum(gold)) == 0 or len(gold[new_nps]) == 0:
                    new_f = 0.0
                else:
                    new_precision = float(sum(gold[new_nps]))/float(len(gold[new_nps]))
                    new_recall = float(sum(gold[new_nps]))/float(sum(gold))
                    new_f = 2.0/(1.0/new_precision+1.0/new_recall) if (new_recall > 0 and new_precision > 0) else 0.0

                if pa == 1:
                    rewards.append(new_f)
                    rewards.append(f)
                else:
                    rewards.append(f)
                    rewards.append(new_f)
            rewards = numpy.array(rewards)
            #rewards -= rewards.max()
            #rewards *= -10

            optimizer.zero_grad()
            #output_softmax = torch.log(output_softmax)
            loss = None
            for i in range(len(output_softmax)):
                pa = path[i]
                gol = gold[i]
                reward_0 = rewards[i*2]
                reward_1 = rewards[i*2+1] 
                max_t = max(reward_0,reward_1)

                reward_0 -= max_t
                reward_1 -= max_t

                reward_0 = reward_0*-10
                reward_1 = reward_1*-10

                #this_reward = -1.0*(res+f)
                #this_reward = -1.0*(res)
                #this_reward = -1.0*f
                #print output_softmax[i]
                #print reward_0,reward_1

                if loss is None:
                    loss = output_softmax[i][0]*reward_0 + output_softmax[i][1]*reward_1
                    #loss = output_softmax[i][pa]*this_reward
                else:
                    loss += (output_softmax[i][0]*reward_0 + output_softmax[i][1]*reward_1)
                    #loss += output_softmax[i][pa]*this_reward

            #loss = loss/len(output_softmax)

            cost += loss.data[0]
            loss.backward()
            optimizer.step()
        print >> sys.stderr, "End epoch",echo,"Cost:", cost

        print "Result for epoch",echo 
        performance.get_performance(test_generater,model) 
if __name__ == "__main__":
    main()
