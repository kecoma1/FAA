from ClasificadorAlgoritmoGenetico import AlgoritmoGenetico
from Datos import Datos

datos = Datos("lentillas.data")
datos = Datos("titanic.csv", allNominal=True)

ag = AlgoritmoGenetico(20, 10, 3, AlgoritmoGenetico.cruceInterReglas, AlgoritmoGenetico.mutacionReglas, 0.2, 0.05, show=True)
ag.entrenamiento(datos.datos, datos.nominalAtributos, datos.diccionario)

print(ag.reglasMejor(datos.diccionario))