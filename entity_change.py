from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'/stanford-corenlp-full-2017-06-09/')
sentence = 'berlin german NA Mr. muller\'s First Stage work, "der lohndrucker" ("the wage dumper"), which was performed in Berlin and elsewhere in 1958, was about an overachieving East German bricklayer. end'
print 'Named Entities:', nlp.ner(sentence)
w_index = 0
word2id = {}
id2word = {}
entity = {'s':'S'}
Entity = {'S':'s'}
etype = {'s':'PERSON'}
def get_entity(fn):
    global w_index
    with open(fn) as f:
        for line in f:
            
            line = line.decode('utf8')
            ss = line.split()
            if len(ss[5:-1]) > 500:
                continue
            el1, el2, e1, e2 = ss[0], ss[1], ss[2], ss[3]
            if len(e1) < 3 or len(e2) <3:
                continue 
            
            for w in ss[5:-1]:
                if not w in word2id:
                    word2id[w] = w_index
                    id2word[w_index] = w
                    w_index += 1
            
            E1 = ' '.join([w.capitalize() for w in e1.split('_')])
            E2 = ' '.join([w.capitalize() for w in e2.split('_')])
            entity[e1] = E1
            entity[e2] = E2
            Entity[E1] = e1
            Entity[E2] = e2

get_entity('train.txt')
get_entity('test.txt')
get_entity('valid.txt')

index = 0
def mark_samples(fn):
    global index
    line_num = 0
    fw = open('T_'+fn, 'wb')
    with open(fn) as f:
        for line in f:
            line_num += 1
            #line = '/m/095fjc /m/0bywvg  berlin german NA Mr. muller\'s First Stage work, "der lohndrucker" ("the wage dumper"), which was performed in Berlin and elsewhere in 1958, was about an overachieving East German bricklayer. end'
            line = line.decode('utf8')
            ss = line.split()
            el1, el2, e1, e2 = ss[0], ss[1], ss[2], ss[3]
            if not (e1 in entity and e2 in entity):
                continue
            E1 = entity[e1]
            E2 = entity[e2]
            ne1 = None
            ne2 = None
            try:
                ner = nlp.ner(' '.join(ss[5:]))
            except:
                continue
            i = 0
            sentence = []
            # print ner
            tmp = []
            ll = ''
            while i < len(ner):
                w = ner[i]
                ww = w[0]
                if w[1] == u'O':
                    if len(tmp):
                        E = ' '.join(tmp)
                        if E in Entity:
                            if Entity[E] == e1:
                                sentence.append('s'+ll)
                                ne1 = 's'+ll
                            elif Entity[E] == e2:
                                sentence.append('e'+ll)
                                ne2 = 'e'+ll
                            elif not ne1 and E.find(E1) != -1:
                                sentence.append('s'+ll)
                                ne1 = 's'+ll
                            elif not ne2 and E.find(E2) != -1:
                                sentence.append('e'+ll)
                                ne2 = 'e'+ll
                        elif not ne1 and E.find(E1) != -1:
                            sentence.append('s'+ll)
                            ne1 = 's'+ll
                        elif not ne2 and E.find(E2) != -1:
                            sentence.append('e'+ll)
                            ne2 = 'e'+ll
                        else:
                            sentence.append(ll)
                        tmp = []
                    ll = ''
                    sentence.append(ww)
                elif w[1] == ll:
                    tmp.append(w[0])
                else:
                    
                    if len(tmp):
                        E = ' '.join(tmp)
                        if E in Entity:
                            if Entity[E] == e1:
                                sentence.append('s'+ll)
                                ne1 = 's'+ll
                            elif Entity[E] == e2:
                                sentence.append('e'+ll)
                                ne2 = 'e'+ll
                            elif not ne1 and E.find(E1) != -1:
                                sentence.append('s'+ll)
                                ne1 = 's'+ll
                            elif not ne2 and E.find(E2) != -1:
                                sentence.append('e'+ll)
                                ne2 = 'e'+ll
                        elif not ne1 and E.find(E1) != -1:
                            sentence.append('s'+ll)
                            ne1 = 's'+ll
                        elif not ne2 and E.find(E2) != -1:
                            sentence.append('e'+ll)
                            ne2 = 'e'+ll
                        else:
                            sentence.append(ll)
                    tmp = [w[0]]
                    ll = w[1]
                    # print 'K:', ll, tmp
                i += 1
            if ne1 and ne2:
                index += 1
                if index % 100 == 0:
                    print index, line_num
                try:
                    fw.write('%s %s %s %s %s %s\n'%(ss[0], ss[1], ne1, ne2, ss[4], ' '.join(sentence)))
                except:
                    pass
            #print ner
            #print e1, e2, E1, E2
#print ne1, ne2, ' '.join(sentence)
                #break
            #if index > 3:
                #break  
            # 
            #print '========='
    fw.close()
mark_samples('U_test.txt')
print index
mark_samples('U_valid.txt')
print index
mark_samples('U_train.txt')
print index