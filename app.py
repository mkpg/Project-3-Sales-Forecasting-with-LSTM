import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.metrics import mean_absolute_error as mean_AbsoluteError,mean_squared_error as MeanSquaredError
import matplotlib.pyplot as plt



df = pd.read_csv('cleaned_sales_data.csv')

grouped_daily_sales = df.groupby('Order_Date')['Net_sales'].sum()
sld_window = 7
dta = grouped_daily_sales.values
x = []
y = []
for i in range(len(dta)-sld_window):
    x.append(dta[i:i+sld_window])
    y.append(dta[i+sld_window])
x_np = np.array(x)
y_np = np.array(y)
split = int(len(x)*(70/100))
x_train,x_test,y_train,y_test = x_np[:split],x_np[split:],y_np[:split],y_np[split:]
# print(x_train.shape,y_train.shape)
scaler_x =  MinMaxScaler()
scaler_y = MinMaxScaler()

scl_x_tr = scaler_x.fit_transform(x_train)
scl_x_te = scaler_x.transform(x_test)

scl_y_tr = scaler_y.fit_transform(y_train.reshape(-1,1))
scl_y_te = scaler_y.transform(y_test.reshape(-1,1))

scl_x_tr = scl_x_tr.reshape(scl_x_tr.shape[0],scl_x_tr.shape[1],1)
scl_x_te = scl_x_te.reshape(scl_x_te.shape[0],scl_x_te.shape[1],1)

model = Sequential([LSTM(50),Dense(1)],trainable = True,name = "sales_forecaster")
# or i can also do model.add(LSTM(50)),model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',metrics=['mae',RootMeanSquaredError(name='rmse')])
trd = model.fit(scl_x_tr,scl_y_tr,batch_size = 3,epochs = 35,shuffle = False,verbose=1)
tst = model.predict(scl_x_te)

pred_act = scaler_y.inverse_transform(tst)
actual_from_y = scaler_y.inverse_transform(scl_y_te)
mae = mean_AbsoluteError(actual_from_y,pred_act)
rmse = np.sqrt(MeanSquaredError(actual_from_y,pred_act))


plt.plot(actual_from_y)
plt.plot(pred_act)
plt.show()