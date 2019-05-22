# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:44:29 2019

@author: marti
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ddf = pd.read_pickle('data.pk1')

#new = ddf[1].str.split(" ", expand = True) 

listStrings = list(ddf[1])

vector = TfidfVectorizer()

response1 = vector.fit(listStrings)
response = vector.fit_transform(listStrings)
print(response)

result = list(vector.get_feature_names())

indices=[]
indices.append(response.indices)

scores=[]
scores.append(response.data)

meanOfScores = np.mean(np.array(scores))