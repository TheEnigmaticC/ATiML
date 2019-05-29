from keras import Sequential
from keras.layers import Dense
import pandas
import pickle

#getting preprocessed data

label_doc=pickle.load(open('label_doc.pkl','rb'))
labellist=pickle.load(open('labellist.pkl','rb'))
text_data=pickle.load(open('data.pkl','rb'))





classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(10000, activation='sigmoid', kernel_initializer='random_normal', input_dim=35069))

#Second  Hidden Layer
classifier.add(Dense(3000, activation='sigmoid', kernel_initializer='random_normal'))

#Output Layer
classifier.add(Dense(201, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])