import pandas as pd
import numpy as np

df = pd.read_csv("ConjuntosDatos/tic-tac-toe.data")
print(df.dtypes.__class__)
print(df.dtypes)
print(df)
tipos = df.dtypes

dicc = {"TLeftSq": {"x": 12}}

a = df.shape

Datos = np.zeros(shape=df.shape)
for i, item in enumerate(df.iterrows()):
    for j, col in enumerate(item[1].items()):
        Datos[i][j] = dicc[col[0]][col[1]]
    print(item)
