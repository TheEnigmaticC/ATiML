import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
import pandas
import pickle

tf.test.is_gpu_available()

#getting preprocessed data
x_train=pickle.load(open('tfidf.pkl','rb'))
y_train=pickle.load(open('y_train.pkl','rb'))

print(x_train)


classifier = keras.Sequential()
#First Hidden Layer
classifier.add(keras.layers.Dense(1000, activation='sigmoid', kernel_initializer='random_normal', input_dim=x_train.shape[1]))

#Second  Hidden Layer
classifier.add(keras.layers.Dense(500, activation='sigmoid', kernel_initializer='random_normal'))

#Output Layer
classifier.add(keras.layers.Dense(201, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

classifier.fit(x_train,y_train,epochs=15,batch_size=1)

tf.test.is_gpu_available()
