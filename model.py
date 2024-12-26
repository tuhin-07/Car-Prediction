'''Car Prediction Model'''

import tensorflow as tf
import seaborn as sns
import pandas as pd
from tensorflow.keras.layers import Normalization , Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("train.csv")
data.head()

sns.pairplot(data[['v.id','on road old',	'on road now',	'years',	'km',	'rating',	'condition',	'economy',	'top speed',	'hp',	'torque',	'current price']],diag_kind='kde')

tensor_data = tf.constant(data)
print(tensor_data)

tensor_data = tf.random.shuffle(tensor_data)
print(tensor_data[:5])

X = tensor_data[:,3:-1]
Y = tensor_data[:,-1]

Train_ratio = 0.8
Val_ratio = 0.1
Test_ratio = 0.1
Dataset_Size = len(X)

X_train = X[:int(Dataset_Size * Train_ratio)]
Y_train = Y[:int(Dataset_Size * Train_ratio)]
print(X_train.shape)
print(Y_train.shape)

X_val = X[int(Dataset_Size * Train_ratio):int(Dataset_Size * (Train_ratio + Val_ratio))]
Y_val = Y[int(Dataset_Size * Train_ratio):int(Dataset_Size * (Train_ratio + Val_ratio))]
print(X_val.shape)
print(Y_val.shape)

X_test = X[int(Dataset_Size * (Train_ratio + Val_ratio)):]
Y_test = Y[int(Dataset_Size * (Train_ratio + Val_ratio)):]
print(X_test.shape)
print(Y_test.shape)

normalizer = Normalization()
normalizer.adapt(X_train)
normalizer(X_train)[:5]

model = tf.keras.Sequential([
                      InputLayer(input_shape=(X.shape[1],)),
                      normalizer,
                      Dense(128,activation='relu'),
                      Dense(128,activation='relu'),
                      Dense(128,activation='relu'),
                      # Dense(16,activation='relu'),
                      Dense(1),
])
# model.add(normalizer)
# model.add(Dense(1))
model.summary()

model.compile(optimizer=Adam(learning_rate=1.0),loss=MeanSquaredError(),metrics=[RootMeanSquaredError()])

history = model.fit(X_train,Y_train, validation_data=(X_val,Y_val), epochs=100,verbose =1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("loss vs epochs")
plt.legend(['train','val_loss'])
plt.show()

plt.plot(history.history["root_mean_squared_error"])
plt.plot(history.history["val_root_mean_squared_error"])
plt.xlabel("epochs")
plt.ylabel("rmse")
plt.title("Model performance")
plt.legend(['train','val'])
plt.show()

model.evaluate(X_test,Y_test)

Y_true = list(Y_test.numpy())

Y_pred = list(model.predict(X_test)[:,0])
print(Y_pred)

ind = np.arange(len(Y_true))
plt.figure(figsize=(40,20))

width = 0.4
plt.bar(ind+width, Y_true, width=width, label='True')
plt.bar(ind, Y_pred, width=width, label='Predicted')

plt.xlabel('Actual vs Predicted Prices')
plt.ylabel('Car Prices')

plt.show()