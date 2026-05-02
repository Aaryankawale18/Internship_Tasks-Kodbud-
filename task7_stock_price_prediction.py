import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# pip install yfinance if you don't have it
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# change this to whatever stock you want to look at
STOCK = "AAPL"
START = "2020-01-01"
END   = "2024-12-31"
PREDICT_DAYS = 30   # how many days ahead to forecast

print(f"Fetching data for {STOCK}...")
df = yf.download(STOCK, start=START, end=END)

if df.empty:
    print("No data found. Check the ticker symbol.")
    exit()

print(f"Got {len(df)} rows")
print(df.tail())

# only using closing price for prediction
prices = df[['Close']].copy()

# plot the raw price history first
plt.figure(figsize=(13, 4))
plt.plot(prices.index, prices['Close'], color='steelblue', linewidth=1.2)
plt.title(f"{STOCK} closing price ({START} to {END})")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("stock_history.png")
plt.show()

# -------------------------------------------------------
# LINEAR REGRESSION
# simple approach: use day number as x, price as y
# -------------------------------------------------------
prices = prices.reset_index()
prices['day_num'] = range(len(prices))

X = prices[['day_num']].values
y = prices['Close'].values

# 80/20 split
cut = int(len(X) * 0.8)
X_train, X_test = X[:cut], X[cut:]
y_train, y_test = y[:cut], y[cut:]

lr = LinearRegression()
lr.fit(X_train, y_train)

y_hat = lr.predict(X_test)

mse = mean_squared_error(y_test, y_hat)
r2  = r2_score(y_test, y_hat)
print(f"\nLinear Regression — MSE: {mse:.2f}  R2: {r2:.4f}")

# predict future days
future_x = np.arange(len(X), len(X) + PREDICT_DAYS).reshape(-1, 1)
future_y  = lr.predict(future_x)

plt.figure(figsize=(13, 4))
plt.plot(prices['day_num'], prices['Close'], label='actual', color='steelblue')
plt.plot(X_test, y_hat, label='lr prediction', color='orange', linewidth=2)
plt.plot(future_x, future_y, label=f'forecast +{PREDICT_DAYS} days',
         color='red', linestyle='--', linewidth=2)
plt.title(f"{STOCK} — Linear Regression Prediction")
plt.xlabel("Day number")
plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lr_prediction.png")
plt.show()

# -------------------------------------------------------
# LSTM (optional but gives better results for time series)
# -------------------------------------------------------
try:
    from tensorflow import keras
    from tensorflow.keras import layers

    print("\nRunning LSTM model...")

    # scale prices to 0-1 range
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices[['Close']].values)

    # use last 60 days to predict the next day
    SEQ = 60

    def make_sequences(data, seq):
        xs, ys = [], []
        for i in range(seq, len(data)):
            xs.append(data[i-seq:i, 0])
            ys.append(data[i, 0])
        return np.array(xs), np.array(ys)

    X_seq, y_seq = make_sequences(scaled, SEQ)
    X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)

    cut2 = int(len(X_seq) * 0.8)
    Xtr, Xte = X_seq[:cut2], X_seq[cut2:]
    ytr, yte = y_seq[:cut2], y_seq[cut2:]

    # model architecture
    model = keras.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=(SEQ, 1)),
        layers.Dropout(0.2),
        layers.LSTM(50),
        layers.Dropout(0.2),
        layers.Dense(25),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    model.fit(Xtr, ytr, epochs=20, batch_size=32, verbose=1)

    # predictions on test portion
    pred_scaled = model.predict(Xte)
    pred_prices = scaler.inverse_transform(pred_scaled)
    actual_prices = scaler.inverse_transform(yte.reshape(-1, 1))

    mse_lstm = mean_squared_error(actual_prices, pred_prices)
    r2_lstm  = r2_score(actual_prices, pred_prices)
    print(f"LSTM — MSE: {mse_lstm:.2f}  R2: {r2_lstm:.4f}")

    # predict next PREDICT_DAYS days step by step
    last_seq = scaled[-SEQ:].reshape(1, SEQ, 1)
    future_lstm = []

    for _ in range(PREDICT_DAYS):
        nxt = model.predict(last_seq, verbose=0)
        future_lstm.append(nxt[0, 0])
        last_seq = np.append(last_seq[:, 1:, :], [[[nxt[0, 0]]]], axis=1)

    future_lstm = scaler.inverse_transform(np.array(future_lstm).reshape(-1, 1))

    # build dates for the future window
    last_date    = pd.to_datetime(prices['Date'].iloc[-1])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                 periods=PREDICT_DAYS, freq='B')

    test_dates = prices['Date'].values[SEQ + cut2:]

    plt.figure(figsize=(13, 4))
    plt.plot(prices['Date'], prices['Close'], label='actual', color='steelblue', alpha=0.7)
    plt.plot(test_dates[:len(pred_prices)], pred_prices,
             label='lstm prediction', color='orange', linewidth=2)
    plt.plot(future_dates, future_lstm,
             label=f'forecast +{PREDICT_DAYS} days', color='red',
             linestyle='--', linewidth=2)
    plt.title(f"{STOCK} — LSTM Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("lstm_prediction.png")
    plt.show()

    model.save("lstm_stock_model.h5")
    print("LSTM model saved.")

except ImportError:
    print("TensorFlow not found, skipping LSTM part. Run: pip install tensorflow")

# summary
print("\n--- Results ---")
print(f"Stock  : {STOCK}")
print(f"Period : {START} to {END}")
print(f"Linear Regression  ->  MSE: {mse:.2f}  |  R2: {r2:.4f}")
try:
    print(f"LSTM               ->  MSE: {mse_lstm:.2f}  |  R2: {r2_lstm:.4f}")
except:
    pass
print("\nTip: change STOCK = 'TSLA' or 'GOOGL' or 'INFY.NS' to try other stocks")
