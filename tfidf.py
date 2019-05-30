# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:44:29 2019

@author: marti
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.model_selection import train_test_split


label_doc=pickle.load(open('label_doc.pkl','rb'))
labellist=pickle.load(open('labellist.pkl','rb'))
print(label_doc[1])
#assign the labels to corresponding document

used_ids=[]
i=-1
label_matrix=np.zeros((19348,201))
for entry in label_doc:
    if int(entry[1]) not in used_ids:
        used_ids.append(int(entry[1]))
        i+=1
        column=labellist.index(entry[0])
        label_matrix[i,column]=1
    else:
        column=labellist.index(entry[0])
        label_matrix[i,column]=1

print(i)



ddf = pd.read_pickle('data.pkl')

listStrings = list(ddf[1])

vector = TfidfVectorizer(min_df=5)
matrix=vector.fit_transform(listStrings)
print(np.shape(matrix))

pickle.dump(matrix,open('tfidf.pkl','wb'))
