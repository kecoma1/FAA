from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd


class Clasificador:

	# Clase abstracta
	__metaclass__ = ABCMeta

	@abstractmethod
	def entrenamiento(self, datosTrain, nominalAtributos, diccionario):
		"""Metodos abstractos que se implementan en cada clasificador concreto

		Args:
				datosTrain: Matriz numpy con los datos de entrenamiento
				nominalAtributos: Array bool con la indicatriz de los atributos nominales
				diccionario: Array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
		"""
		pass

	@abstractmethod
	def clasifica(self, datosTest, nominalAtributos, diccionario):
		"""Esta funcion debe ser implementada en cada clasificador concreto. Devuelve un numpy array con las predicciones

		Args:
				datosTest: Matriz numpy con los datos de validación
				nominalAtributos: Array bool con la indicatriz de los atributos nominales
				diccionario: Array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
		"""
		pass

	def error(self, datos, pred):
		"""Obtiene el numero de aciertos y errores para calcular la tasa de fallo

		Args:
				datos: Matriz numpy con los datos de entrenamiento
				pred: Predicción
		"""
		error = 0.0
		errores = 0

		for i in range(datos.datos.shape[0]):
			if datos[i] != pred[i]:
				errores += 1

		return (errores/datos.datos.shape[0])*100

	def validacion(self, particionado, dataset, clasificador, seed=None):
		""" Realiza una clasificacion utilizando una estrategia de particionado determinada

		Args:
				particionado: Estrategia de particionado
				dataset: Dataset
				clasificador: Clasificador a usar
		"""

	# Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
	# - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
	# y obtenemos el error en la particion de test i
	# - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
	# y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
		pass

	##############################################################################


class ClasificadorNaiveBayes(Clasificador):
	"""Clase que define un clasificador el cuál usa el algoritmo de
		Naive Bayes. Las probabilidades se guardan en 3 atributos:
		* Un Array numpy con la frecuencia de cada clase.
		* Un array numpy donde se guardan las probabilidades a priori.
		* Una matriz Numpy el cual tiene la siguiente forma: 
			Nº filas = Nº clases
			Nº columnas = Nº Atributos
		De esta forma se guarda en la fila X y la columna Y un array numpy
		con las probabilidades condicionales de la clase y el atributo.
		Por ejemplo:
			P(Yi | X), siendo X la clase e Yi el atributo/dato/columna.
			Y es una array numpy que contiene cada probabilidad condicional
			de esa columna.
		Por ejemplo:
				Atr1								Atr2					...
		C1		[P(Atr1[0]|C1), P(Atr1[1]|C1) ...]	[P(Atr2[0]|C1), ...]	...
		C2		[P(Atr1[0]|C2), P(Atr1[1]|C2) ...]	[P(Atr2[0]|C2), ...]	...
		...		...									...

		Estos mismos atributos también se crean pero aplicando la corrección de LaPlace (CLP).
	"""

	def __init__(self, numClases, numAtributos):
		"""Constructor.

		Args:
			numClases: Número valores de la clase en el dataset.
			numAtributos: Número de atributos en el dataset.
		"""
		self.clasesFreq = None
		self.pClases = {}
		self.pCondicionales = None

		# Atributos con la corrección de la Laplace
		self.CLPclasesFreq = None
		self.CLPpClases = {}
		self.CLPpCondicionales = None


	def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
		"""Método que calcula las probabilidades siguiendo el algoritmo

		Args:
				datostrain: Datos a usar para entrenar el modelo.
				atributosDiscretos: Columnas de la matriz de datos
				donde estos son discretos.
				diccionario: 
		"""
		# Calculamos la probabilidad de las clases
		self.calculaPClases(datostrain, diccionario)

		# Calculamos las probabilidades condicionadas
		self.calculaPCondicionadas(datostrain, atributosDiscretos, diccionario)
	

	# TODO: implementar
	def clasifica(self, datostest, atributosDiscretos, diccionario):
		pass

	def calculaPClases(self, datostrain, diccionario):
		"""Método para calcular la probabilidad de las clases (frecuencia).

		Args:
			datostrain (Matriz Numpy): Datos a comprobar.
			diccionario (dict): Diccionario con los atributos y las clases.
		"""
		# Obtenemos la frecuencia de los valores en las columnas
		_, freq = list(np.unique(datostrain[:,-1], return_counts=True))
		total = sum(freq)
		self.pClases = np.array([i/total for i in freq])


	def calculaPCondicionadas(self, datostrain, atributosDiscretos, diccionario):
		# Calculamos el número de filas y columnas de la matriz 
		# con las probabilidades condicionadas
		nCols = datostrain.shape[1]-1
		nRows = len(list(np.unique(datostrain[-1])))

		# Construimos la matriz
		pCondicionales = []
		for i in range(nRows):
			pCondicionales.append([])
			for n in range(nCols):
				columnas = len(list(np.unique(datostrain[n])))
				pCondicionales[i].append(np.zeros(columnas))

		# Calculamos las probabilidades condicionales
		cols = np.unique(datostrain[:,-1])
		for i, classVal in enumerate(cols): # Por cada valor de la clase
			for n, atr in enumerate(diccionario.items()): # Por cada atributo/columna
				
				if n == len(diccionario)-1: # En caso de que lleguemos a la clase
					break
				elif not atributosDiscretos[n]: # En caso de que no sea discreto calculamos la probabilidad continua
					pass # calcula probabilidad_continua

				# Declaramos el array interno de cada atributo_i|clase
				pCondicionales[i][n] = np.zeros(len(atr[1]))
				for atrValue in range(len(atr[1])): # Por cada valor en el atributo

					for row in datostrain: # Comprobamos cada la fila
						if row[-1] == classVal and row[n] == atrValue:
							pCondicionales[i][n][atrValue] += 1
					pCondicionales[i][n][atrValue] /= self.freq[i]
