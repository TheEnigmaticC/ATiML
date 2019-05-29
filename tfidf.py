# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:44:29 2019

@author: marti
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ddf = pd.read_pickle('data.pkl')

#new = ddf[1].str.split(" ", expand = True) 

listStrings = list(ddf[1])

vector = TfidfVectorizer()
matrix=vector.fit_transform(listStrings)