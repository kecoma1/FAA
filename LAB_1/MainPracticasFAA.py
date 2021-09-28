from Datos import Datos
import pandas as pd
import EstrategiaParticionado

# Cargamos los datasets
print("Cargando dataset tic-tac-toe ...")
dataset1=Datos('ConjuntosDatos/tic-tac-toe.data')

print("Cargando dataset german ...")
dataset2=Datos('ConjuntosDatos/german.data')

# Cargamos los datos en un dataframe
print("Cargando dataframe tic-tac-toe ...")
df1 = pd.read_csv("ConjuntosDatos/tic-tac-toe.data")

print("Cargando dataframe german ...")
df2 = pd.read_csv("ConjuntosDatos/german.data")

# TICTACTOE
# Comparamos los datos con el dataset construido
print("[TIC-TAC-TOE] DATOS:\n", dataset1.datos[:3])
print(df1.head(3))
print("[TIC-TAC-TOE] NOMINAL ATRIBUTOS:\n", dataset1.nominalAtributos, "\n")

print("[TIC-TAC-TOE] DICCIONARIO ITEMS:")
for dicc in dataset1.diccionario.items():
    print(dicc)

print("\n[TIC-TAC-TOE] DICCIONARIO ENTERO:\n"+str(dataset1.diccionario))

# Probamos las estrategias de particionado
print("[TIC-TAC-TOE] VALIDACIÓN SIMPLE")
vc = EstrategiaParticionado.ValidacionSimple(10, 5) 
vc.creaParticiones(dataset1.datos)

for i in range(5):
    print()
    print("***Test***:", vc.particiones[i].indicesTest)
    print("***Train***:", vc.particiones[i].indicesTrain)

print("[TIC-TAC-TOE] VALIDACIÓN CRUZADA")
vc = EstrategiaParticionado.ValidacionCruzada(10) 
vc.creaParticiones(dataset1.datos)

for i in range(10):
    print()
    print("***Test***:", vc.particiones[i].indicesTest)
    print("***Train***:", vc.particiones[i].indicesTrain)

print("\n-----------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------\n")
# GERMAN
# Comparamos los datos con el dataset construido
print("[GERMAN] DATOS:\n", dataset2.datos[:3])
print(df2.head(3))
print("[GERMAN] NOMINAL ATRIBUTOS:\n", dataset2.nominalAtributos, "\n")

print("[GERMAN] DICCIONARIO ITEMS:")
for dicc in dataset2.diccionario.items():
    print(dicc)

print("\n[GERMAN] DICCIONARIO ENTERO:\n"+str(dataset2.diccionario))

# Probamos las estrategias de particionado
print("[GERMAN] VALIDACIÓN SIMPLE - GERMAN")
vc = EstrategiaParticionado.ValidacionSimple(10, 5) 
vc.creaParticiones(dataset1.datos)
for i in range(5):
    print()
    print("***Test***:", vc.particiones[i].indicesTest)
    print("***Train***:", vc.particiones[i].indicesTrain)

print("[GERMAN] VALIDACIÓN CRUZADA - GERMAN")
vc = EstrategiaParticionado.ValidacionCruzada(10) 
vc.creaParticiones(dataset1.datos)
for i in range(10):
    print()
    print("***Test***:", vc.particiones[i].indicesTest)
    print("***Train***:", vc.particiones[i].indicesTrain)