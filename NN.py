import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
import pandas
import pickle

#getting preprocessed data
x_train=pickle.load(open('tfidf.pkl','rb'))
y_train=pickle.load(open('y_train.pkl','rb'))

print(x_train)


classifier = keras.Sequential()
#First Hidden Layer
classifier.add(keras.layers.Dense(10000, activation='sigmoid', kernel_initializer='random_normal', input_dim=x_train.shape[1]))

#Second  Hidden Layer
classifier.add(keras.layers.Dense(3000, activation='sigmoid', kernel_initializer='random_normal'))

#Output Layer
classifier.add(keras.layers.Dense(201, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

classifier.fit(x_train,y_train,epochs=3,batch_size=100)