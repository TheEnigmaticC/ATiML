import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import pickle

#getting labellist
labellist=pickle.load(open('labellist.pkl','rb'))
#getting training and testdata
#shape x_train/x_test: rows represent documents, columns represent words from all documents, cells represent tfidf score
#shape y_train/y_test: rows represent documents, columns represent labels, cells indicitade whether or not a label is assigned to a document
x_train=pickle.load(open('tfidf.pkl','rb'))
y_train=pd.DataFrame(pickle.load(open('y_train.pkl','rb')))
x_test=pickle.load(open('x_test.pkl','rb'))
y_test=pd.DataFrame(pickle.load(open('y_test.pkl','rb')))


print(x_train)
print(y_train)

print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_test))
print(np.shape(y_test))


#define OnevsRestClassifier with Multinomial Naive Bayes Estimator
orbc = OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))


#train one classifier for each label
for label in range(200):
    #train the model on training data
    orbc.fit(x_train, y_train[label])
    #calculate testing accuracy using accuracy_score function
    prediction=orbc.predict(x_test)
    print(prediction)
    #print accuracy for each label
    print('Test accuracy is {}'.format(accuracy_score(y_test[label], prediction)))
