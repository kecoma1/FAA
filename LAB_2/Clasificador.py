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

		En el caso de los atributos continuos se guarda una lista con 2 elementos, la media y la varianza:
		...		Atr1
		CX		[media, varianza] # En este mismo orden
	"""

	def __init__(self):
		"""Constructor.

		Args:
			numClases: Número valores de la clase en el dataset.
			numAtributos: Número de atributos en el dataset.
		"""
		self.Freq = None
		self.pClases = {}
		self.pCondicionales = None

		# Atributos con la corrección de la Laplace
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
		self.calculaPClases(datostrain)

		# Calculamos las probabilidades condicionadas
		self.calculaPCondicionales(datostrain, atributosDiscretos, diccionario)


	# TODO: implementar
	def clasifica(self, datostest, atributosDiscretos, diccionario):
		pass


	def calculaPClases(self, datostrain):
		"""Método para calcular la probabilidad de las clases (frecuencia).

		Args:
			datostrain (Matriz Numpy): Datos a comprobar.
		"""
		# Obtenemos la frecuencia de los valores en las columnas
		_, self.freq = list(np.unique(datostrain[:,-1], return_counts=True))
		total = sum(self.freq)
		self.pClases = np.array([i/total for i in self.freq])


	def calculaPCondicionales(self, datostrain, atributosDiscretos, diccionario):
		"""Método para calcular las probabilidades condicionales.

		Args:
			datostrain (numpy.array): Matriz con los datos.
			atributosDiscretos (list): Columnas de la matriz de datos
			diccionario (dict): Diccionario con información sobre los atributos y la clase.
		"""
		# Calculamos el número de filas y columnas de la matriz 
		# con las probabilidades condicionadas
		nCols = datostrain.shape[1]-1
		nRows = len(list(np.unique(datostrain[:,-1])))

		# Construimos la matriz
		self.pCondicionales = []
		self.CLPpCondicionales = []
		for i in range(nRows):
			self.pCondicionales.append([])
			self.CLPpCondicionales.append([])
			for n in range(nCols):
				columnas = len(list(np.unique(datostrain[n])))
				self.pCondicionales[i].append(np.zeros(columnas))
				self.CLPpCondicionales[i].append(np.zeros(columnas))

		# Calculamos las probabilidades condicionales
		for classIndex in range(nRows): # Por cada valor de la clase
			# Obtenemos el valor específico de la clase a comprobar
			if atributosDiscretos[-1]:
				classValue =list(list(diccionario.items())[-1][1].values())[classIndex]
			else:
				classValue = list(np.unique(datostrain[:,-1]))[classIndex]
			for atrIndex in range(nCols): # Por cada atributo/columna
				# Query para obtener unicamente las filas con la clase deseada
				rowsToCheck = np.where(datostrain[:,-1]==classValue)
				if not atributosDiscretos[atrIndex]: # En caso de que no sea discreto calculamos la probabilidad continua
					self.calculaPCondicionalContinua(atrIndex, classIndex, datostrain)
				else:
					self.calculaPCondicionalDiscreta(atrIndex, classIndex, rowsToCheck, datostrain, diccionario)


	def calculaPCondicionalDiscreta(self, atrIndex, classIndex, rowsToCheck, datostrain, diccionario):
		"""Método para calcular la probabilidad condicional de un atributo discreto.

		Args:
			atrIndex (int): Índice del atributo.
			classIndex (int): Índice de la clase.
			rowsToCheck (numpy.array): Array con el índice de las filas que contienen el valor de la clase correcto.
			datostrain (numpy.array): Matriz con los datos.
			diccionario (dict): Diccionario con información sobre los atributos y la clase.
		"""
		# Declaramos el array interno de cada atributo_i|clase
		uniqueInCol = len(list(diccionario.values())[atrIndex].values())
		self.pCondicionales[classIndex][atrIndex] = np.zeros(uniqueInCol)
		self.CLPpCondicionales[classIndex][atrIndex] = np.zeros(uniqueInCol) # Array laplace
		freqs = np.zeros(uniqueInCol)

		for atrValue in range(uniqueInCol): # Por cada valor en el atributo
			for row in datostrain[rowsToCheck]: # Comprobamos cada la fila
				if row[atrIndex] == atrValue:
					self.pCondicionales[classIndex][atrIndex][atrValue] += 1
			freqs[atrValue] = self.pCondicionales[classIndex][atrIndex][atrValue]
			self.pCondicionales[classIndex][atrIndex][atrValue] /= self.freq[classIndex]

		# En caso de que haya un valor a 0 en las frecuencias, aplicamos la corrección de laplace
		if 0 in self.pCondicionales[classIndex][atrIndex]:
			for i in range(uniqueInCol):
				# Para calcular la correción de laplace añadimos 1
				self.CLPpCondicionales[classIndex][atrIndex][i] = freqs[i] + 1
				self.CLPpCondicionales[classIndex][atrIndex][i] /= (self.freq[classIndex] + len(self.CLPpCondicionales[classIndex][atrIndex]))
		else:
			self.CLPpCondicionales[classIndex][atrIndex] = np.copy(self.pCondicionales[classIndex][atrIndex])


	def calculaPCondicionalContinua(self, atrIndex, classIndex, datostrain):
		"""Método para calcular la probabilidad continua.

		Args:
			atrIndex (int): Índice del atributo.
			classIndex (int): Índice de la clase.
			datostrain (numpy.array): Matriz con los datos.
		"""
		self.pCondicionales[classIndex][atrIndex] = np.zeros(2)

		# Guardamos la media y la varianza
		self.pCondicionales[classIndex][atrIndex][0] = np.mean(datostrain[:,atrIndex])
		self.pCondicionales[classIndex][atrIndex][1] = np.var(datostrain[:,atrIndex])