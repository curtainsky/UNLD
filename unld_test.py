import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import time
# import sys
def one_epoch(b2s=1000, s2b=100, s2s=40, epoch_i=0):
    print '=*=*=*=*=*=*=*=*=*='
    print b2s,s2b,s2s,epoch_i
    print '-------------------'
    clsnum = [100000, 6000, 5000, 4000, 3000, 2000]
    wst = 2000
    EPOCH_NUM = 1500
    wse = 40
    embnum = wst+wse*(len(clsnum)-1)
    low_index = 0.78
    # if int(sys.argv[1]) == 2:
    #     low_index = 0.78
    wlist = range(0, embnum)
    entity = [0,1,2, 3,4,5]
    for i in range(len(entity)):
        entity[i] += embnum
    stype = [[[wst+wse*0, wst+wse*1], [entity[0],entity[0+3]], clsnum[1]],
             [[wst+wse*1, wst+wse*2], [entity[0],entity[1+3]], clsnum[2]],
             [[wst+wse*1, wst+wse*2], [entity[0],entity[0+3]], clsnum[2]],
             [[wst+wse*2, wst+wse*3], [entity[0],entity[2+3]], clsnum[3]],
             [[wst+wse*2, wst+wse*3], [entity[1],entity[1+3]], clsnum[3]],
             [[wst+wse*3, wst+wse*4], [entity[1],entity[2+3]], clsnum[4]],
             [[wst+wse*3, wst+wse*4], [entity[0],entity[0+3]], clsnum[4]],
             [[wst+wse*3, wst+wse*4], [entity[2],entity[0+3]], clsnum[4]],
             [[wst+wse*4, wst+wse*5], [entity[0],entity[0+3]], clsnum[5]],
             [[wst+wse*4, wst+wse*5], [entity[1],entity[2+3]], clsnum[5]]]

    for i in range(len(stype)):
        stype[i].append(i+1)
    enl = []
    for x in stype:
        enl.append(x[1])
    embnum += len(entity)
    print embnum

    def generate(s,e, enl):
        ss = random.sample(range(s,e), 5)
        ee = random.sample(wlist, 15)
        return ss + ee + enl

    # train data
    train = [generate(0,wst, random.sample(enl,1)[0]) for i in range(clsnum[0])]
    label = [0] * clsnum[0]
    # miss train data
    real_label = label[:]
    for parm in stype:
        for n in range(b2s):
            train.append(generate(parm[0][0], parm[0][1], parm[1]))
            label.append(0)
            real_label.append(parm[3])

    # other type
    for i in range(len(stype)):
        parm = stype[i]
        for n in range(parm[2]):
            train.append(generate(parm[0][0], parm[0][1], parm[1]))
            label.append(parm[3])
            real_label.append(parm[3])
        # 0 miss
        for n in range(s2b):
            train.append(generate(0, wst, random.sample(enl,1)[0]))
            label.append(parm[3])
            real_label.append(0)

        # other miss
        for j in range(len(stype)):
            if i != j:
                rparm = stype[j]
                for n in range(int(random.random()*s2s)):
                    train.append(generate(rparm[0][0], rparm[0][1], rparm[1]))
                    label.append(parm[3])
                    real_label.append(rparm[3])

    data = train
    print len(train), len(label), len(real_label)
    
    train_var = Variable(torch.LongTensor(data)).cuda()
    train_label = Variable(torch.LongTensor(label)).cuda()

    def modify(indices, train_label, racc):
        label = train_label.data.tolist()
        tmp = {}
        # print racc
        for i in range(len(indices)):
            p,l = indices[i], label[i]
            if l == 0 and p != l:
                if p in racc:
                    train_label[i] = p
                    tmp.setdefault(p, [0])[0] += 1
        print "Convert:", tmp

    def calc_acc(indices, train_label, real_label):
        lacc,racc = {}, {}
        lrec,rrec = {}, {}
        predict = indices
        for i in range(len(predict)):
            r,l,p = real_label[i], train_label[i], predict[i]

            td = lacc.setdefault(p, [0, 0])
            if p == l: td[1] += 1
            else: td[0] += 1

            td = racc.setdefault(p, [0, 0])
            if p == r: td[1] += 1
            else: td[0] += 1

            td = lrec.setdefault(l, [0, 0])
            if p == l: td[1] += 1
            else: td[0] += 1

            td = rrec.setdefault(l, [0, 0])
            if p == r: td[1] += 1
            else: td[0] += 1
        tt = 0
        ll = 0
        RACC = []
        for i in range(len(stype)+1):
            if not i in lacc:
                print i, 0, 0, '|', 0, 0, '|', 0, 0
                tt += 1
                continue
            la,ra = lacc[i],racc[i]
            lr,rr = lrec[i],rrec[i]
            if float(ra[1])/(ra[0]+ra[1]) < low_index:
                ll +=1
            else:
                RACC.append(i)
            print '%d %0.4f %0.4f | %0.4f %0.4f | %d %d'%(i, 
                                                      float(la[1])/(la[0]+la[1]), 
                                                      float(ra[1])/(ra[0]+ra[1]), 
                                                      float(lr[1])/(lr[0]+lr[1]), 
                                                      float(rr[1])/(rr[0]+rr[1]),
                                                      la[0]+la[1],
                                                      rr[0]+rr[1])
        if tt > 1 or ll > 6:
            return False
        return RACC

    class AttentionModel(nn.Module):
        def __init__(self, w_embd_len=10, seq_out_len=9, e_embd_len=5, word_len=20):
            super(AttentionModel, self).__init__()
            self.word_linear = nn.Linear(e_embd_len, word_len)
            self.seq_linear = nn.Linear(w_embd_len, seq_out_len)
            self.cat_linear = nn.Linear(seq_out_len+word_len, 1)
            
        def forward(self, seq, word):
            nw = F.relu(self.word_linear(word))
            batch_size = seq.size()[0]
            seq_len = seq.size()[1]
            seq_l = F.relu(self.seq_linear(seq))
            wl = nw.view(batch_size, 1, -1).repeat(1, seq_len, 1)
            sw = torch.cat((seq_l, wl), dim=2)
            zz = F.relu(self.cat_linear(sw))
            att = F.softmax(zz.view(batch_size, seq_len), dim=1)
            ns = att.view(batch_size, seq_len, 1).mul(seq)
            return ns

    class AModel(nn.Module):
        def __init__(self, out_len = 10, emdb_len = 40,
                           seq_out_len = 50,
                           e_linear_len = 20):
            super(AModel, self).__init__()
            self.embedding = nn.Embedding(embnum, emdb_len)
            self.atten = AttentionModel(emdb_len, seq_out_len, emdb_len*2)
            self.ilinear = nn.Linear(emdb_len, emdb_len)
            self.olinear = nn.Linear(emdb_len, out_len)

        def forward(self, seq):
            se = self.embedding(seq)
            w = torch.cat([se[:,-2], se[:,-1]], dim=1)
            se = self.atten(se, w)
            sl = F.relu(self.ilinear(se))
            o = F.relu(self.olinear(sl)).sum(1)
            return o

    def run(train_var, train_label, epoch_num=1500, model = None):
        times = 0
        #re_times = 0
        Recheck = False
        sacc = 0
        bacc = 0
        lr = 0.01
        while (1):
            torch.cuda.manual_seed(201)
            if type(model) == type(None):
                model = AModel(len(stype)+1).cuda()
            loss_function = nn.CrossEntropyLoss()
            
            optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-5)
            accurate=0
            tlen = len(label)
            #acc_check = 0
            s_time = time.time()
            for epoch in range(1, epoch_num):
                torch.cuda.empty_cache()
                model.zero_grad()
                tag_scores = model(train_var)
                values, indices = tag_scores.max(1)
                accurate = indices.eq(train_label).data.float().sum()
                loss = loss_function(tag_scores, train_label)
                loss.backward()
                optimizer.step()
                if epoch%100==0:
                    print 'epoch:',epoch,'loss:',loss.data[0],'acc:',accurate/tlen, 'time:', time.time() - s_time
                    if epoch == 100:
                        sacc = accurate/tlen
                    
                    s_time = time.time()
                    if accurate/tlen < 0.7:
                        Recheck = True
                        break
                    racc = True
                    #if epoch == 800:
                    #    indices = indices.data.tolist()
                    #    indices_list.append([indices, train_label.data.tolist()])
                    #    racc = calc_acc(indices, tmp_label, real_label)
                    if (not racc) or accurate/tlen < 0.9 and (epoch == 800 and accurate/tlen-sacc < 0.002 or (abs(accurate/tlen - bacc) < 1e-11)):
                        Recheck = True
                        break
                    bacc = accurate/tlen
            
            if times < 10 and Recheck:
                Recheck = False
                times += 1
                model = None
                print 'Recheck!',times
                continue
            else:
                if times >= 10:
                    return None, None
                else:
                    break

        return indices, model

    

    indices_list = []
    tmp_label = label
    times = 0
    enum = EPOCH_NUM
    model = None
    for n in range(15):
        indices, model = run(train_var, train_label, enum, model)
        if type(indices) == type(None):
            print '%s/15'%n
            model = None
            enum = 2000
            continue
        enum = 1000
        # check = False
        print '--------------'
        indices = indices.data.tolist()
        indices_list.append([indices, train_label.data.tolist()])
        racc = calc_acc(indices, tmp_label, real_label)
        print 'racc:', racc
        if racc:
            times+=1
        else:
            model = None
            enum = 2000
            continue
        modify(indices, train_label, racc)
        tmp_label = train_label.data.tolist()
        np.save("data/big-%d-%d-%d-%d-%d.npy"%(b2s, s2b, s2s, epoch_i, i), np.array([indices, tmp_label, real_label]))
        print '=============='
        if times > 10:
            return True

if __name__ == '__main__':
    # nohup python unld_test.py > log/unld.log 2>&1 & 
    param_list = [[1000, 100, 50], [1000, 300, 50], [1000, 500, 50], [1000, 500, 100], 
                  [5000, 100, 50], [5000, 300, 50], [5000, 500, 50], [5000, 500, 100], 
                  [8000, 100, 50], [8000, 300, 50], [8000, 500, 50], [8000, 500, 100]]

    # param[0] the number of other categories sample in NA
    # param[1] the number of NA in other categories
    # param[2] the number of other categories in one category
    for param in param_list:
        for i in range(6):
            if one_epoch(param[0], param[1], param[2], i):
                break

