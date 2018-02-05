#encoding:utf8
from __future__ import unicode_literals, print_function, division
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import sys
ipyout = sys.stdout

reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = ipyout

from io import open
#import unicodedata
#import string
#import re
import random
#import torch.autograd as autograd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

# torch.cuda.set_device(0)

import time

WORD_FRE = 10
print ("Loading data....")
test_fn = 'RawData/en/s_re_test.txt'
train_fn = 'RawData/en/s_re_train.txt'
valid_fn = 'RawData/en/s_re_valid.txt'


word2id = {}
id2word = {}
w_index = 1
wordnum = {}

def load_data(fn):
    global w_index
    global ew_index
    global e_index
    tmp_relations = []
    with open(fn) as f:
        for line in f:
            #line = re.sub('\t', ' ', line)
            line = line.decode('utf8').lower()
            ss = line.split()
            e1, e2 = ss[2], ss[3]
            if e2[0] == 's':
                continue
            for w in ss[5:-1]:
                if not w in word2id:
                    word2id[w] = w_index
                    id2word[w_index] = w
                    w_index += 1
                wordnum.setdefault(w, [0])[0] += 1
            tmp_relations.append( ['%s %s'%(e1, e2), ss[4], ss[5:-1]])
    return tmp_relations


def load_data_tmp(fn):
    tmp_relations = []
    with open(fn) as f:
        for line in f:
            line = line.decode('utf8')
            ss = line.split()
            e1, e2 = ss[2], ss[3]
            tmp_relations.append( ['%s %s'%(e1, e2), ss[4], ss[5:-1]])
    return tmp_relations

valid_relations = load_data(valid_fn)
train_relations = load_data(train_fn)
test_relations = load_data(test_fn)
relation2id = {}
with open('RawData/en/relation2id.txt') as f:
    for line in f:
        rel, id = line.strip().split('\t')
        relation2id[rel] = int(id)
print ("valid_relations:", len(valid_relations), 
       "train_relations:", len(train_relations), 
       "test_relations:", len(test_relations), 
       #"entity_word2id:", len(eword2id), 
       "word2id:", len(word2id),
       #"entity2id:", len(entity2id),
       "relation2id:", len(relation2id))


label_change = {}
label_sss = {}
for x in train_relations:
    label = x[1]
    if label == u'P885':
        print ("1", x)
    if label in label_sss:
        label_sss[label] += 1
    else:
        label_sss[label] = 1
large_num = 0
sum_s = 0
value = 500
for x in sorted(label_sss.items(), key=lambda x:x[1], reverse=True):
    if x[1] > value:
        large_num+= 1 
    else:
        sum_s += x[1]

print (large_num, sum_s)
large_index = 0
for x in sorted(label_sss.items(), key=lambda x:x[1], reverse=True):
    if x[1] > value:
        label_change[x[0]] = large_index
        large_index += 1
    else:
        label_change[x[0]] = large_num
    
print (len(label_change), large_num)

class Relation:
    def __init__(self):
        self.data = []
    def add(self, ritem):
        label = None
        e1, e2 = ritem[0].split()
        stag = word2id[e1]
        etag = word2id[e2]
        relatin = [word2id[w] for w in ritem[2] if wordnum[w][0] >= WORD_FRE]
        label = label_change[ritem[1]]
        if ritem[1] in label_change:
            label = label_change[ritem[1]]
        if label != None:
            self.data.append([[stag], [etag], relatin, label])
            # print ([stag, etag, relatin, label])

valid_data = Relation()
train_data = Relation()
test_data = Relation()

for ritem in train_relations:
    train_data.add(ritem)
for ritem in valid_relations:
    valid_data.add(ritem)
for ritem in test_relations:
    test_data.add(ritem)

print (len(train_data.data), len(valid_data.data), len(test_data.data))
print ("Generate GPU Variable....")
def generate_mini_batch(data, batch_size=1000):
    mini_batches = []
    batch = []
    data.sort(key=lambda x: len(x[2]))
    slen = 0
    elen = 0
    for i in range(len(data)):
        batch.append(data[i])
        tmp = len(data[i][0])
        if tmp > slen: slen = tmp
        tmp = len(data[i][1])
        if tmp > elen: elen = tmp
        if len(batch) == batch_size and i:
            mini_batches.append([batch, slen, elen, len(batch[-1][2])])
            slen = 0
            elen = 0
            batch = []
    if len(batch):
        mini_batches.append([batch, slen, elen, len(batch[-1][2])])
    return mini_batches

def padding_batches(blocks):
    mini_batches = []
    for mini_batch in blocks:
        si = []
        ei = []
        ri = []
        label = []
        slen,elen,rlen = mini_batch[1:]
        if rlen < 6:
            rlen = 6
        for record in mini_batch[0]:
            # print (record)
            s,e,r,l = record
            si.append([0]*(slen-len(s))+s)
            ei.append([0]*(elen-len(e))+e)
            ri.append([0]*(rlen-len(r))+r)
            label.append(l)
        mini_batches.append([Variable(torch.LongTensor(si)).cuda(), 
                             Variable(torch.LongTensor(ei)).cuda(), 
                             Variable(torch.LongTensor(ri)).cuda(), 
                             Variable(torch.LongTensor(label)).cuda()])
    return mini_batches

train_batches = padding_batches(generate_mini_batch(train_data.data))
valid_batches = padding_batches(generate_mini_batch(valid_data.data))
test_batches = padding_batches(generate_mini_batch(test_data.data))
print (len(train_batches), len(valid_batches), len(test_batches))

class AttentionModel(nn.Module):
    def __init__(self, w_embd_len=10, seq_out_len=9, e_embd_len=5):
        super(AttentionModel, self).__init__()
        self.seq_linear = nn.Linear(w_embd_len, seq_out_len)
        self.cat_linear = nn.Linear(seq_out_len+e_embd_len, 1)
        self.word_linear = nn.Linear(e_embd_len, e_embd_len)
        
    def forward(self, seq, word):
        batch_size = seq.size()[0]
        seq_len = seq.size()[1]
        seq_l = F.relu(self.seq_linear(seq))
        wl = word.view(batch_size, 1, -1).repeat(1, seq_len, 1)
        sw = torch.cat((seq_l, wl), dim=2)
        zz = F.relu(self.cat_linear(sw))
        att = F.softmax(zz.view(batch_size, seq_len), dim=1)
        ns = att.view(batch_size, seq_len, 1).mul(seq)
        nw = F.relu(self.word_linear(word))
        return ns, nw
    
class CNNModel(nn.Module):
    def __init__(self, 
                 w_embd_num=592286, 
                 w_embd_len=100, 
                 seq_out_len=200, 
                 #e_embd_num=42511, 
                 #e_embd_len=100, 
                 e_linear_len=40, 
                 output=30,
                 cnn_kernel_num=10,
                 cnn_k_gram=5,
                 cnn_k_size=6,
                 pooling_size=5):
        super(CNNModel, self).__init__()
        self.char_embed = nn.Embedding(w_embd_num, w_embd_len, padding_idx=0)
        self.e_linear = nn.Linear(w_embd_len*2, e_linear_len)
        self.atts1 = AttentionModel(w_embd_len, seq_out_len, e_linear_len)
        self.cnn2d = nn.Conv2d(1, cnn_kernel_num, (cnn_k_gram, cnn_k_size))
        self.pooling = nn.AvgPool2d((1, pooling_size))
        rlen = (w_embd_len-cnn_k_size+1)/pooling_size
        self.soft_linear = nn.Linear(int(rlen), output)
        
    def forward(self, sw, ew, seq):
        
        re = self.char_embed(seq)
        se = self.char_embed(sw)
        ee = self.char_embed(ew)
        
        w = torch.cat((se, ee), dim=2)
        w = F.relu(self.e_linear(w))
        
        re,w = self.atts1(re, w)
        s = re.size()
        rl = self.cnn2d(re.view(s[0],1,s[1],s[2]))
        rl = self.pooling(rl).sum(2).sum(1)
        
        output = F.relu(self.soft_linear(rl))
        return output

print ("Build model....")
torch.cuda.manual_seed(201)
weight = [1.0]*(large_num+1)
weight[0] = 0.8

weight = Variable(torch.FloatTensor(weight)).cuda()
model = None
loss_function = None
optimizer = None

while (1):
    print (len(word2id), large_num+1)
    model = CNNModel(w_embd_num=len(word2id)+1,
                     #e_embd_num=len(eword2id)+1,
                     output=large_num+1).cuda() 

    loss_function = nn.CrossEntropyLoss(weight)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    mlen = len(train_batches)

    s_time = time.time()
    ss_time = s_time
    loss_value = 0.0
    index = 0
    accurate = 0.0
    acc_test = 0.0
    acc_valid = 0.0
    for i in random.sample(range(mlen), mlen):
        torch.cuda.empty_cache()
        batch = train_batches[i]
        model.zero_grad()
        # print ("Batch:",i)
        tag_scores = model(batch[0], batch[1], batch[2])
        #
        values, indices = tag_scores.max(1)
        accurate += indices.eq(batch[-1]).data.float().sum()
        # result.append([indices.data, batch[-1]])
        loss = loss_function(tag_scores, batch[-1])
        loss.backward()
        loss_value += loss.data[0]
        optimizer.step()
        if index % 500 == 0 and index:
            print ("Check:", "batch:", index, "time:",time.time()-s_time)
            #s_time = time.time()
        index += 1
    for batch in valid_batches:
        torch.cuda.empty_cache()
        tag_scores = model(batch[0], batch[1], batch[2])
        values, indices = tag_scores.max(1)
        acc_valid += indices.eq(batch[-1]).data.float().sum()
    print ("Check:",
           accurate/len(train_data.data), 
           acc_valid/len(valid_data.data))
    if acc_valid/len(valid_data.data) > 0.76:
        print ("pass")
        break

print ("Start traning....")
result = []
vmax = 0.0
for epoch in range(1000):
    s_time = time.time()
    ss_time = s_time
    loss_value = 0.0
    index = 0
    accurate = 0.0
    acc_test = 0.0
    acc_valid = 0.0
    for i in random.sample(range(mlen), mlen):
        torch.cuda.empty_cache()
        batch = train_batches[i]
        model.zero_grad()
        tag_scores = model(batch[0], batch[1], batch[2])
        #
        values, indices = tag_scores.max(1)
        accurate += indices.eq(batch[-1]).data.float().sum()
        # result.append([indices.data, batch[-1]])
        loss = loss_function(tag_scores, batch[-1])
        loss.backward()
        loss_value += loss.data[0]
        optimizer.step()
        if index % 500 == 0 and index:
            print ("epoch:",epoch, "batch:", index, "time:",time.time()-s_time)
            s_time = time.time()
        index += 1

    for batch in valid_batches:
        torch.cuda.empty_cache()
        tag_scores = model(batch[0], batch[1], batch[2])
        values, indices = tag_scores.max(1)
        acc_valid += indices.eq(batch[-1]).data.float().sum()

    for batch in test_batches:
        torch.cuda.empty_cache()
        tag_scores = model(batch[0], batch[1], batch[2])
        values, indices = tag_scores.max(1)
        acc_test += indices.eq(batch[-1]).data.float().sum()
    
    print ("epoch:",epoch, "batch:", index, 'loss:', loss_value, 
           "acc:", accurate/len(train_data.data), 
           "valid:",acc_valid/len(valid_data.data),
           "test:",acc_test/len(test_data.data), "time:",time.time()-ss_time)
    if vmax < acc_valid/len(valid_data.data) or epoch%50 == 0:
        torch.save(model.state_dict(), 'modelstore/en_t_matt-cnn_08_1/en_t_matt-cnn_08_1-%d.save' % epoch)
        vmax = acc_valid/len(valid_data.data)