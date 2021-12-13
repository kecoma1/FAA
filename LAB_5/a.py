from ClasificadorAlgoritmoGenetico import AlgoritmoGenetico
from Datos import Datos

datos = Datos("lentillas.data")
datos = Datos("titanic.csv", allNominal=True)

ag = AlgoritmoGenetico(20, 100, 2, AlgoritmoGenetico.cruceIntraReglas, AlgoritmoGenetico.mutacionReglas , 0.1, 0.1, show=True)
ag.entrenamiento(datos.datos, datos.nominalAtributos, datos.diccionario)
print(ag.clasifica(datos.datos[:10], datos.nominalAtributos, datos.diccionario))