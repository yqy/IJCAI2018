#coding=utf8
import argparse
import random
import numpy as np
import properties_loader
import sys

#parse arguments
DIR="/mnt/sshd/qingyu/"
parser = argparse.ArgumentParser(description="Experiemts for ZP resolution (by qyyin)\n")
parser.add_argument("-raw_data",default = DIR+"data/", type=str, help="raw_data")
parser.add_argument("-embedding_data",default = DIR+"embedding/sogou.word2vec.100d.txt", type=str, help="raw embedding data")
parser.add_argument("-data",default = DIR+"zp/", type=str, help="saved vectorized data")
parser.add_argument("-props",default = "./properties/prob.sample", type=str, help="properties")
parser.add_argument("-reduced",default = 0, type=int, help="reduced")
parser.add_argument("-gpu",default = 0, type=int, help="GPU number")

parser.add_argument("-random_seed",default=12345,type=int,help="random seed")
#parser.add_argument("-random_seed",default=12345,type=int,help="random seed")

args = parser.parse_args()

args = parser.parse_args()

random.seed(args.random_seed)
np.random.seed(args.random_seed)

nnargs = properties_loader.read_pros(args.props)
for item in nnargs.items():
    print >> sys.stderr, item
    print item
print
