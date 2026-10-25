import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam


np.random.seed(42)
num_days = 1000


price = 100
closes = []
for _ in range(num_days):
    price += np.random.normal(0, 1.2)
    closes.append(price)

df = pd.DataFrame(closes, columns=['Close'])
df['Open'] = df['Close'].shift(1) + np.random.normal(0, 0.5, num_days)
df['High'] = df[['Open','Close']].max(axis=1) + np.random.uniform(0, 1.5, num_days)
df['Low'] = df[['Open','Close']].min(axis=1) - np.random.uniform(0, 1.5, num_days)
df['Volume'] = np.random.randint(10000, 80000, num_days)
df = df.dropna().reset_index(drop=True)



X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']
                      


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


input_shape = [4]

earlystop = callbacks.EarlyStopping(

min_delta= 0.001,

patience= 60,

restore_best_weights=True,

)

omar_model = keras.Sequential([

layers.Dense(units= 64 , activation= 'relu' , input_shape=input_shape),
layers.Dense(units= 32 , activation= 'relu'),
layers.Dense(units= 1 ),

])

omar_model.compile(

optimizer='adam',
loss='mae',

)

history = omar_model.fit(
 X_train, y_train,



batch_size = 32,

 validation_data=(X_valid, y_valid),

epochs = 120,

callbacks = [earlystop],




)

history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));



predicted_price = omar_model.predict(X_valid)

print("closed price")
print("___________________________________________________________")
print(predicted_price)
print("___________________________________________________________")