from ClasificadorAlgoritmoGenetico import AlgoritmoGenetico
from Datos import Datos

datos = Datos("lentillas.data")

ag = AlgoritmoGenetico(20, 1000, 10, AlgoritmoGenetico.cruceInterReglas, AlgoritmoGenetico.mutacionReglas , 0.1, 0.1, show=True)
ag.entrenamiento(datos.datos, datos.nominalAtributos, datos.diccionario)