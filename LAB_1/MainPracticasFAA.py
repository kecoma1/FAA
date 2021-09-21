from Datos import Datos
import pandas as pd
import EstrategiaParticionado

# Cargamos los datasets
dataset1=Datos('ConjuntosDatos/tic-tac-toe.data')
dataset2=Datos('ConjuntosDatos/german.data')

# Cargamos los datos en un dataframe
df1 = pd.read_csv("ConjuntosDatos/tic-tac-toe.data")
df2 = pd.read_csv("ConjuntosDatos/german.data")

# TICTACTOE
# Comparamos los datos con el dataset construido
print("DATOS:\n", dataset1.datos[:3])
print(df1.head(3))
print("NOMINAL ATRIBUTOS:\n", dataset1.nominalAtributos, "\n")

print("DICCIONARIO ITEMS:")
for dicc in dataset1.diccionario.items():
    print(dicc)

print("\nDICCIONARIO ENTERO:\n"+str(dataset1.diccionario))

# Probamos las estrategias de particionado
vc = EstrategiaParticionado.ValidacionSimple(10, 5) 
vc.creaParticiones(dataset1.datos)

for i in range(5):
    print()
    print("***Test***:", vc.particiones[i].indicesTest)
    print("***Train***:", vc.particiones[i].indicesTrain)

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
print("DATOS:\n", dataset2.datos[:3])
print(df2.head(3))
print("NOMINAL ATRIBUTOS:\n", dataset2.nominalAtributos, "\n")

print("DICCIONARIO ITEMS:")
for dicc in dataset2.diccionario.items():
    print(dicc)

print("\nDICCIONARIO ENTERO:\n"+str(dataset2.diccionario))

# Probamos las estrategias de particionado
print("VALIDACIÓN SIMPLE")
vc = EstrategiaParticionado.ValidacionSimple(10, 5) 
vc.creaParticiones(dataset1.datos)
for i in range(5):
    print()
    print("***Test***:", vc.particiones[i].indicesTest)
    print("***Train***:", vc.particiones[i].indicesTrain)

print("VALIDACIÓN CRUZADA")
vc = EstrategiaParticionado.ValidacionCruzada(10) 
vc.creaParticiones(dataset1.datos)
for i in range(10):
    print()
    print("***Test***:", vc.particiones[i].indicesTest)
    print("***Train***:", vc.particiones[i].indicesTrain)