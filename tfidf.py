# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:44:29 2019

@author: marti
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ddf = pd.read_pickle('data.pk1')

listStrings = list(ddf[1])

vector = TfidfVectorizer(min_df=5)
matrix=vector.fit_transform(listStrings)
print(matrix[0])
