from ClasificadorAlgoritmoGenetico import AlgoritmoGenetico
from Datos import Datos

datos = Datos("lentillas.data")
datos = Datos("tic-tac-toe.data")
datos = Datos("titanic.csv", allNominal=True)

ag = AlgoritmoGenetico(30, 50, 5, AlgoritmoGenetico.cruceInterReglas, AlgoritmoGenetico.mutacionReglas, 0.2, 0.05, show=True)
ag.entrenamiento(datos.datos, datos.nominalAtributos, datos.diccionario)
print(ag.reglasMejor(datos.diccionario))