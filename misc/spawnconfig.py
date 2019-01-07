import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib import pyplot as plt


def extra_vars(df):
    df = df.copy()
    df['halitePerShip'] = df['availableHaliteOnInitShip'] / (df['numberOfShipsTotal'] + 1)
    df['turnsRemaining'] = df['totalTurns'] - df['turnOnInit']
    df['shipRatio'] = (df['numberOfShipsPlayer'] / df['numberOfShipsEnemies']).fillna(0.5)
    df['shipDiff'] = df['numberOfShipsPlayer'] - (df['numberOfShipsEnemies'] - df['numberOfShipsPlayer']) / (df['players'] - 1)
    return df


def read(fp: str):
    df = (pd.read_csv(fp, sep=',').pipe(extra_vars))
    df = (df.loc[(df['didCollideEndGame'] == 1) &
                 (df['isDropoff'] == 0) &
                 (df['turnOnInit'] >= 100)
                 ]
          .groupby(['players', 'map'])
          .apply(model)
          # .to_json('spawnconfig.json', orient='index')
          )
    return df


def model(df: pd.DataFrame):
    return ols("""returned ~ turnsRemaining + availableHaliteOnInitShip""", data=df).fit().rsquared_adj


df = read(fp='181223_ships.csv')
print(df)
# mod = model(df=df)
# fig = plt.figure(figsize=(10, 6))
# fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
#
# fig = plt.figure(figsize=(10, 6))
# fig = sm.graphics.plot_regress_exog(mod, "availableHaliteRatioOnInitShip", fig=fig)
