# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:44:29 2019

@author: marti
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ddf = pd.read_pickle('data.pk1')

#new = ddf[1].str.split(" ", expand = True) 

listStrings = list(ddf[1])

vector = TfidfVectorizer()
vector.fit(listStrings)