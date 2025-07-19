# 'Close_t-1','Close_t-2','Close_t-3','Close_t-4','Close_t-5','Volume','High','Low','Open'

import training_model
import yfinance as yf
import joblib
import pandas as pd

ticker = 'GOOGL'
data = yf.download(ticker , period='10d' ,auto_adjust=True)
data.columns = [col[0] for col in data]
data.dropna()

if len(data) < 6 : 
    raise ValueError("Not enough data for a prediction !!! ")

close_t5_to_t1 = list(data['Close'].iloc[-6:-1][::-1])


latest_row = data.iloc[-1]
scaler = joblib.load('volume_scaler.pkl')
volume = latest_row['Volume']
volume = latest_row['Volume']
scaled_volume = scaler.transform([[volume]])[0][0]
high = latest_row['High']
low = latest_row['Low']
open_price = latest_row['Open']


featues = close_t5_to_t1 + [scaled_volume, high , low , open_price]

featute_names = ['Close_t-1','Close_t-2','Close_t-3','Close_t-4','Close_t-5','Volume','High','Low','Open']

X_input=pd.DataFrame([featues], columns=featute_names)


model = joblib.load('model.pkl')
predicted_close = model.predict(X_input)

print(f"Predicted closing price for the next trading day: ${predicted_close[0]:.2f}")

#output : Predicted closing price for the next trading day: $184.92