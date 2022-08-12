import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU


# Reading Data
df = pd.read_csv("gafgyt_danmini_doorbell_train.csv")
df_test = pd.read_csv("gafgyt_danmini_doorbell_test.csv")
# Keeping only features columns for the train set
df_features = df.loc[:, df.columns != "target"]
print(f"Shape of the train set: {df_features.shape}")
y_train = df.target
# Keeping only features for the test set
df_features_test = df_test.loc[:, df_test.columns != "target"]
y_test = df_test.target
# Applying the normalization on the train then test set
scaler = MinMaxScaler()
df_features = scaler.fit_transform(df_features)
df_features_test = scaler.transform(df_features_test)

# Implementation of the Autoencoder Model
# input from df_features, dense64, leakyrelu, dense32, leakyrelu, dense16, tanh 
input = Input(shape=df_features.shape[1:])
enc = Dense(64)(input)
enc = LeakyReLU()(enc)
enc = Dense(32)(enc)
enc = LeakyReLU()(enc)
# latent space with tanh
latent_space = Dense(16, activation="tanh")(enc)

dec = Dense(32)(latent_space)
dec = LeakyReLU()(dec)
dec = Dense(64)(dec)
dec = LeakyReLU()(dec)

dec = Dense(units=df_features.shape[1], activation="sigmoid")(dec)
# init model
autoencoder = Model(input, dec)
# compile model
autoencoder.compile(optimizer = "adam",metrics=["mse"],loss="mse")
# train model
autoencoder.fit(df_features, df_features, epochs=50, batch_size=32, validation_split=0.25)
encoder = Model(input, latent_space)
# predict on test set
test_au_features = encoder.predict(df_features_test)
print(test_au_features.shape)