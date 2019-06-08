import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from keras.models import load_model
from tensorflow.python.keras.callbacks import TensorBoard
import pandas
import pickle
from time import time


tf.test.is_gpu_available()

#getting preprocessed data
x_train=pickle.load(open('tfidf.pkl','rb'))
y_train=pickle.load(open('y_train.pkl','rb'))
x_test=pickle.load(open('x_test.pkl','rb'))
y_test=pickle.load(open('y_test.pkl','rb'))


classifier = keras.Sequential()
#First Hidden Layer
classifier.add(keras.layers.Dense(1500, activation='sigmoid', kernel_initializer='random_normal', input_dim=x_train.shape[1]))

#Second  Hidden Layer
classifier.add(keras.layers.Dense(500, activation='sigmoid', kernel_initializer='random_normal'))

#Output Layer
classifier.add(keras.layers.Dense(201, activation='sigmoid', kernel_initializer='random_normal'))

#Instanciate Tensorboard
tensorboard=TensorBoard(log_dir='logs/{}'.format(time()))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss=tf.keras.losses.BinaryCrossentropy(), metrics=['binary_accuracy'])

classifier.fit(x_train,y_train,epochs=50,batch_size=10, callbacks=[tensorboard])


classifier.save('classifier.h5')

y_pred=classifier.predict_proba(x_test)
print(y_pred[0])
print(y_test[0])

classifier=load_model('classifier.h5')

classifier.fit(x_train,y_train,epochs=100,batch_size=10, callbacks=[tensorboard])