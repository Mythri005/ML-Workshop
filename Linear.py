# import keras from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.random.randn(100, 1)
#print(l)

y = 2*x + 0.005
#print(y)

model = Sequential()
model.add(Dense(2, input_dim = 1))
model.add(Dense(1))
model.compile(optimizer = SGD(learning_rate=0.001), loss = 'mean_squared_error')

model.fit(x,y, epochs =100)

output = model.predict(x)
#print(output)

plt.scatter(x,y, label = 'Original data')
plt.plot(x, output, label = 'Predicted data')
plt.show()