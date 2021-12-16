from ClasificadorAlgoritmoGenetico import AlgoritmoGenetico
from Datos import Datos

datos = Datos("lentillas.data")
datos = Datos("titanic.csv", allNominal=True)

ag = AlgoritmoGenetico(20, 100, 10, AlgoritmoGenetico.cruceInterReglas, AlgoritmoGenetico.mutacionReglas, 0.05, 0.05, show=True)
ag.entrenamiento(datos.datos, datos.nominalAtributos, datos.diccionario)
print(ag.clasifica(datos.datos[:10], datos.nominalAtributos, datos.diccionario))

