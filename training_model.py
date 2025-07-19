import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


google = yf.download("GOOGL",period="5y",interval='1d',auto_adjust=True)
df = google.copy()


df.columns = [col[0] for col in df]
df.info()
df.describe()


# scalling the volume featuer 
scaler = StandardScaler()
df['Volume'] = scaler.fit_transform(google['Volume'])
joblib.dump(scaler, 'volume_scaler.pkl')

# Lag Features (or Time-Shifted Features)
df['Close_t-1'] = df['Close'].shift(1)
df['Close_t-2'] = df['Close'].shift(2)
df['Close_t-3'] = df['Close'].shift(3)
df['Close_t-4'] = df['Close'].shift(4)
df['Close_t-5'] = df['Close'].shift(5)


df.dropna(inplace=True)


x_train = df[['Close_t-1','Close_t-2','Close_t-3','Close_t-4','Close_t-5','Volume','High','Low','Open']]
y_train = df['Close']


#splitting 80% data for the testing purpose
split_index = int(len(x_train) * 0.8)
X_train, X_test = x_train.iloc[:split_index], x_train.iloc[split_index:]
Y_train, y_test = y_train.iloc[:split_index], y_train.iloc[split_index:]


model = LinearRegression()
model.fit(X_train,Y_train)
joblib.dump(model, 'model.pkl')

w = model.coef_
b = model.intercept_
print(f"Parameters = w :{w} , b :{b}")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)

print(f"Mean sqaured error : {mse:}")

plt.figure(figsize=(8,4))


plt.plot(Y_train.values, label='Actual Close Price', color='blue')

y_temp_pred = model.predict(X_train)

plt.plot(y_temp_pred, label='Predicted Close Price', color='orange')

plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Time (days)')
plt.ylabel('Stock Price in $')
plt.legend()
plt.grid(True)
plt.show()