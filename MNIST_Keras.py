import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.datasets import mnist
from keras.utils import to_categorical        #For encoding the categorical values
from keras.callbacks import ModelCheckpoint

#Load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, X_test.shape)
#plt.imshow(X_train[0])
#plt.show()

#Flatten the image
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
print(X_train.shape, X_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)
#print(y_train[0])

#Normalization
#print(X_train.max(), X_test.max())
X_train = X_train/255
X_test = X_test/255
print(X_train.max(), X_test.max())

model = Sequential()
model.add(Dense(10, input_dim=784, activation='softmax'))  #Every neuron must take 784 pixels i.e., th complete image
model.compile(optimizer = SGD(learning_rate=0.0001), 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
cB = ModelCheckpoint('my_model.keras', monitor = 'val_loss', save_best_only ='True', mode = 'min')

res = model.fit(X_train, y_train, 
                epochs = 30, 
                batch_size = 32,
                validation_split=0.2,
                callbacks = [cB])
#batch size update the parameters after every 32 rows

model.evaluate(X_test, y_test)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(res.history['loss'], label='Training loss')
plt.plot(res.history['val_loss'], label='Validation loss')  
#validation loss is dataset is not there hence while model building 20% data is used for validation
plt.legend()

plt.subplot(1,2,2)
plt.plot(res.history['accuracy'], label='Accuracy')
plt.plot(res.history['val_accuracy'], label='Validation accuracy') 
plt.legend()
plt.show()