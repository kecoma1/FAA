from ClasificadorAlgoritmoGenetico import AlgoritmoGenetico
from Datos import Datos

datos = Datos("lentillas.data")

ag = AlgoritmoGenetico(10, 10, 5, 0,0,0)
ag.entrenamiento(datos, datos.nominalAtributos, datos.diccionario)