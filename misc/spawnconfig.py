import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense

from matplotlib import pyplot as plt


def extra_vars(df):
    df = df.copy()
    df['halitePerShip'] = df['availableHaliteOnInitShip'] / (df['numberOfShipsTotal'] + 1)
    df['turnsRemaining'] = df['totalTurns'] - df['turnOnInit']
    df['shipRatio'] = (df['numberOfShipsPlayer'] / df['numberOfShipsEnemies']).fillna(0.5)
    df['shipDiff'] = df['numberOfShipsPlayer'] - (df['numberOfShipsEnemies'] - df['numberOfShipsPlayer']) / (
            df['players'] - 1)
    return df


def read(fp: str):
    df = (pd.read_csv(fp, sep=',').pipe(extra_vars))
    df = (df.loc[(df['didCollideEndGame'] == 1) &
                 (df['isDropoff'] == 0) &
                 (df['turnOnInit'] >= 100) &
                 (df['players'] == 4) &
                 (df['map'] == '48x48')
                 ]
        # .groupby(['players', 'map'])
        # .apply(model)
        # .to_json('spawnconfig.json', orient='index')
        )
    return df


def model(df: pd.DataFrame):
    return ols("""returned ~ turnsRemaining + availableHaliteOnInitShip + halitePerShip""", data=df).fit()


def get_rmse(df: pd.DataFrame):
    return model(df).rsquared_adj


df = read(fp='181223_ships.csv')

X = df.loc[:, ['haliteStart', 'availableHaliteOnInitShip', 'availableHaliteRatioOnInitShip', 'turnOnInit', 'totalTurns',
               'numberOfShipsPlayer', 'numberOfShipsEnemies', 'numberOfShipsTotal', 'halitePerShip', 'turnsRemaining',
               'shipRatio', 'shipDiff']].values
# X = df.loc[:, ['availableHaliteOnInitShip', 'turnsRemaining']].values
y = df.loc[:, 'returned'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(12, input_dim=12, kernel_initializer='normal', activation='relu', ))
# Adding the output layer
classifier.add(Dense(output_dim=1, kernel_initializer='normal', activation='relu'))
# Compiling Neural Network
classifier.compile(loss='mean_squared_error', optimizer='adam')
# Fitting our model
classifier.fit(X_train, y_train, batch_size=10, epochs=150)

print(abs(classifier.predict(X_test) - y_test).mean())
plt.plot(classifier.history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Average difference:
# 12 parameters = 1206 halite
# 2 parameters = 1216 halite

# mod = model(df=df)
# fig = plt.figure(figsize=(10, 6))
# fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
#
# fig = plt.figure(figsize=(10, 6))
# fig = sm.graphics.plot_regress_exog(mod, "availableHaliteOnInitShip", fig=fig)
