
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

#load imported data
label_doc=pickle.load(open('label_doc.pkl','rb'))
labellist=pickle.load(open('labellist.pkl','rb'))
print(label_doc[1])

#make a matrix of labels (document x labelvector)
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



#import text data
ddf = pd.read_pickle('data.pkl')
ddf.columns=['id','text']

#get ids from text data and delete the text data for the ones that are not in our labels
doc_ids=list(ddf['id'])
doc_ids=list(map(int,doc_ids))
no_labels=list(set(doc_ids)-set(used_ids))
ddf[~ddf.id.isin(no_labels)]
listStrings = list(ddf['text'])



x_train, x_test, y_train, y_test=train_test_split(listStrings,label_matrix,test_size=0.2)

vector = TfidfVectorizer(min_df=5)
matrix=vector.fit_transform(x_train)
print(np.shape(matrix))

pickle.dump(matrix,open('tfidf.pkl','wb'))
pickle.dump(vector,open('vectorizer.pkl','wb'))
pickle.dump(y_train,open('y_train.pkl','wb'))
print(matrix)
