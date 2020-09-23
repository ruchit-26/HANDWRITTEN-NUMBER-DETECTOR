#import tensorflow
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
print('Using TensorFlow version', tf.__version__)
#The Dataset import MNIST
from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#Shapes of imported Arrays
print('x_ train shape',x_train.shape)
print('y_ train shape',y_train.shape)
print('x_ test shape',x_test.shape)
print('y_ test shape',y_test.shape)
#Plot an image example
from matplotlib import pyplot as plt
%matplotlib inline
plt.imshow(x_train[1],cmap='binary')
plt.show()
#Display Labels
print(y_train[1])
#Encoding labels for ex->5=0000010000
from tensorflow.keras.utils import to_categorical
y_train_encoded=to_categorical(y_train)
y_test_encoded=to_categorical(y_test)
#validate shapes
print('y_train_encoded shape:',y_train_encoded.shape)
print('y_test_encoded shape:',y_test_encoded.shape)
#Display Encoded Labels
print(y_train_encoded[0])
#Unrolling N-dimensional array to vectors
import numpy as np
x_train_reshaped=np.reshape(x_train,(60000,784))
x_test_reshaped=np.reshape(x_test,(10000,784))
print('x_train_reshaped:',x_train_reshaped.shape)
print('x_test_reshaped:',x_test_reshaped.shape)
#Display Pixel values
print(set(x_train_reshaped[0]))
#Data Normalisation
x_mean=np.mean(x_train_reshaped)
x_std=np.std(x_train_reshaped)
epsilon=1e-10
x_train_norm=(x_train_reshaped-x_mean)/(x_std+epsilon)
x_test_norm=(x_test_reshaped-x_mean)/(x_std+epsilon)
#Display Normalised Pixel values
print(x_train_norm[0])
#creating a Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential([
    Dense(128,activation='relu',input_shape=(784,)),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
])
#compiling the model
model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    
)
model.summary()
#Training the model
model.fit(x_train_norm,y_train_encoded,epochs=3)
#Evaluating the model
loss,accuracy=model.evaluate(x_test_norm,y_test_encoded)
print(accuracy*100)
# Predctions on Test Set
preds=model.predict(x_test_norm)
print('shape of preds',preds.shape)
#PLOTTING THE RESULT
plt.figure(figsize=(12,12))

start_index=0
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    pred=np.argmax(preds[start_index+i])
    gt=y_test[start_index+i]
    col='g'
    if pred!=gt:
        col='r'
    plt.xlabel('i={},pred={},gt={}'.format(start_index+i,pred,gt),color=col)
    plt.imshow(x_test[start_index+i],cmap='binary')
    plt.show()

