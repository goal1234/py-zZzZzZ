# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 09:19:14 2018

@author: gogoing
"""
import numpy as np
import re
import jieba
import pandas as pd
import jieba.analyse

def Fenci(test, stop_dict):
    res = []
    s = int(0)
    for art in test:
        art = art.replace("-", "").replace("\x01", "").replace("—", "")
        art = re.sub("。|、|”|“|:|,|，|（|）|：","", art)
        test_fc=[i for i in jieba.cut(art)]
        res.append(" ".join(test_fc))
        print(s)
        s += 1
    return res
    

def Score(test, dictin, dict_beta, dict_point, position):
    position = int(position)
    base_score = []
    beta_score = []  #程度词
    base_point = []  #观点词
    start = int(0)
    t = test.split(" ")
    if len(t) < position: return int(2)
    for i in t:
        if i in dictin.keys():
            # 多了5次的循环
            if position !=0:
                if t.index(i) >= position:
                    index = t.index(i)
                    t1 =t[index - position:index]
                    for j in t1:
                        if j in dict_beta.keys():
                            beta_score.append(dict_beta[j])
                        if j in dict_point.keys():
                            base_point.append(dict_point[j])
                            
            base_score.append(dictin[i])
        start += 1
        
    base_score = sum(base_score, 0)/len(t)
    base_point = sum(base_point, 0)/len(t)
    
    if beta_score == []:
        beta_score = 1
    elif len(beta_score) > 10:
        beta_score = 3.5
    else:
        beta_score = max(np.array(beta_score))
            
    p = (base_score+base_point)*beta_score
    p = round(p, 2)
    return p




class MakeScore():
    def __init__(self):
        self.dict_neg = {}
        self.dict_pos = {}
        self.dict_point = {}
        self.dict_beta = {}
        self.fc = []
        
        
    def p_neg(self, x):
        return Score(x, self.dict_neg, self.dict_beta, self.dict_point, 3)
        
    def p_pos(self, x):
        return Score(x, self.dict_pos, self.dict_beta, self.dict_point, 3)
    
    
    def textrank_neg(self, x):
        a = " ".join(jieba.analyse.textrank(x))
        return Score(a, self.dict_neg, self.dict_beta, self.dict_point, 0)
    
    def textrank_pos(self, x):
        a = " ".join(jieba.analyse.textrank(x))
        return Score(a, self.dict_pos, self.dict_beta, self.dict_point, 0)
    
    
    def Make_it(self):
        print(" 占比计算中...")
        nn = list(map(self.p_neg, self.fc))
        pp = list(map(self.p_pos, self.fc))
        
        print("textrank计算中....\n..\n.")
        s1 = list(map(self.textrank_neg, self.fc))    
        s2 = list(map(self.textrank_pos, self.fc))
        
        aa  = pd.DataFrame([np.array(nn), np.array(pp)])
        aa = aa.T
        aa['all'] = aa.sum(axis= 1)
        aa['c'] = aa[1]/aa[0]
        aa['textrankneg'] = s1
        aa['textrankpos'] = s2
        print('分数计算完毕')
        return aa



def __panduan(neg, pos, al, c, textrankneg, textrankpos):
    tk = 0
    if al >= 0.6:
        if c >= 1.2:
            res = 20
        elif c >= 1:
            res = 15
        else:
            res = 5
    elif al >= 0.3:
        if c >=1.5:
            res = 15
        elif c >= 1:
            res = 12
        else:
            res = 5
    else:
        if c >= 2:
            res = 15
        elif c>= 1:
            res = 10
        else:
            res = 5
    
    if textrankpos > textrankneg:
        tk = 5
    
    p = tk*0.2 + res*0.8
    return p

def __label(x):
    if x >= 10:
        label = '正面'
    elif x <= 4.5:
        label = '负面'
    else:
        label = '中性'
    return label

def get_label(aa):
    p = map(__panduan,aa[0], aa[1],aa['all'], aa['c'],aa['textrankneg'], aa['textrankpos'])
    p = list(p) 
    
    aa['res'] = p 
    aa['label'] = aa['res'].apply(__label)
    
    return aa


def store(aa, fc, name, path = r'E:\job\正负分类\dict\output'):
    import os
    p = os.getcwd()    
    os.chdir(path)
    base = 'score' + name + '.xlsx'
    out = name + '.xlsx'
    pd.DataFrame(aa).to_excel(base, index =False)
    fc['正负面'] = aa['label']
    pd.DataFrame(fc).to_excel(out, index = False)
    os.chdir(p)
    print("------完成------")    