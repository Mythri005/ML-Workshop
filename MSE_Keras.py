from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

california = fetch_california_housing()
#print(type(california))
#print(california['data'].shape)
data = pd.DataFrame(california.data, columns=california.feature_names)
#print(type(data))
#print(data.shape)
#print(data.columns)
#print(california.target)

col = data.columns
print(col)
X = california.data
y = california.target

#data['price'] = california.target
#print(data.head)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
#X_train, X_test, y_train, y_test = train_test_split(data, data['price'], test_size=0.2, random_state = 0)

#Normalize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train the model
Regressor = LinearRegression()
Regressor.fit(X_train, y_train)

y_pred = Regressor.predict(X_test)

#No accuracy score because the the labels are not binary
MSE = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", MSE)

plt.scatter(X_test[:, 0], y_test, label='Original data', color='Blue')
plt.plot(X_test[:, 0], y_pred, label='Predicted data', color = 'Red')
plt.show()


