import gensim 
import json 
import numpy as np 
from gensim.models import Word2Vec
import os  
import random
import re
with open("sentences/ebu3b.json") as f1:
    ebu3b=json.load(f1)
with open("sentences/ap_m.json") as f2:
    ap_m=json.load(f2)
with open("sentences/label_ebu3b.json") as f1:
    label_ebu3b=json.load(f1)
with open("sentences/label_ap_m.json") as f2:
    label_ap_m=json.load(f2)
a=[]
b=[]
for i in label_ebu3b:
    a+=label_ebu3b[i]
    
for j in label_ap_m:
    b+=label_ap_m[j]

    
a=set(a)
b=set(b)
num=0
public_set=a&b
for i in b:
    if i in a:
        num+=1
for i in label_ebu3b:
    if i in label_ap_m:
        label_ebu3b[i]=list(set(label_ebu3b[i]+label_ap_m[i]))
for i in label_ap_m:
    if i not in label_ebu3b:
        label_ebu3b[i]=label_ap_m[i]
label=label_ebu3b

sentences=ebu3b
#filter the sentences
filter_words=["VendorGivenName",""]
new_sentences=[]
for i in sentences:
    new_line=[]
    for j in i:
        if j not in filter_words and j in label:
                new_j=re.sub("[0-9]","#",j)
                new_line.append(new_j)
    new_sentences.append(new_line)
sentences=new_sentences

model = Word2Vec(min_count=1,
                     window=1,
                     size=30,  #dim=10
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=3)
                     
model.build_vocab(sentences)
model.train(sentences,total_examples=model.corpus_count, epochs=10000, report_delay=1)

f=open('building_wordvec_30d_ebu3b.txt','w')
for word,word_obj in model.wv.vocab.items():
    f.write(word+" "+" ".join(list([str(i) for i in model.wv[word]]))+"\n")
f.close()
