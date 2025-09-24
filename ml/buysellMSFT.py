import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Download historical data for Microsoft (MSFT)
ticker = 'MSFT'
start_date = '2020-01-01'
end_date = '2023-01-01'

df = yf.download(ticker, start=start_date, end=end_date)

# Feature engineering: momentum and returns
window = 10
df['Momentum'] = df['Close'] - df['Close'].shift(window)
df['Return'] = df['Close'].pct_change(periods=window)
df.dropna(inplace=True)

# Target variable
# Targets: Buy(-1) if return > 5%, Sell(1) if < -5%, else Hold(0) for stricter threshold
df['Target'] = 0
df.loc[df['Return'] > 0.05, 'Target'] =-1
df.loc[df['Return'] < -0.05, 'Target'] = 1

# Prepare features and labels
features = df[['Momentum', 'Close']].values
labels = tf.keras.utils.to_categorical(df['Target'] + 1, num_classes=3)  # Map -1,0,1 to 0,1,2

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, shuffle=False, test_size=0.3)

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(features.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Predict signals
df['PredictedSignal'] = np.argmax(model.predict(features), axis=1) - 1  # Map back to -1,0,1

# Visualize buy/sell signals
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='Close', color='blue')
plt.scatter(df.index[df['PredictedSignal'] == 1], df['Close'][df['PredictedSignal'] == 1], marker='^', color='g', label='Buy', s=90)
plt.scatter(df.index[df['PredictedSignal'] == -1], df['Close'][df['PredictedSignal'] == -1], marker='v', color='r', label='Sell', s=90)
plt.title(f'Trained ML Buy/Sell Signals for {ticker}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
