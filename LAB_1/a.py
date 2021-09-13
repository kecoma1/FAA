import pandas as pd

df = pd.read_csv("ConjuntosDatos/tic-tac-toe.data")
print(df.dtypes.__class__)
print(df.dtypes)
print(df)
tipos = df.dtypes

dicc = {}
for line in df.iterrows():
    print(line[1])
    a = line[1].values
    b = line[1].keys()
    line[1].
    print(a)
