from numpy.core.fromnumeric import shape
from Clasificador import Clasificador
from scipy.stats import norm
import numpy as np
import math


class ClasificadorKNN(Clasificador):

	def __init__(self, K):
		self.K = K

	def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
		# Normalizamos los datos
		self.datosNormalizados = self.normalizarDatos(datostrain, atributosDiscretos)

		# Calculamos la probabilidad de las clases
		self.calculaPClases(self.datosNormalizados)

	def clasifica(self, datostest, atributosDiscretos, diccionario):
		prediccionesClases = []
		testNormalizados = self.normalizarDatos(datostest, atributosDiscretos)
		for rowTest in testNormalizados: # Por cada fila en el test
			prediccionesClases.append(self.clasificaFila(rowTest))
		return prediccionesClases

	def calculaPClases(self, datostrain):
		"""Método para calcular la probabilidad de las clases (frecuencia).

		Args:
			datostrain (Matriz Numpy): Datos a comprobar.
		"""
		# Obtenemos la frecuencia de los valores en las columnas
		_, self.freq = np.unique(datostrain[:,-1], return_counts=True)
		self.N = sum(self.freq)
		self.pClases = np.array([i/self.N for i in self.freq])

	def calcularMediasDesv(self, datos, nominalAtributos):
		"""Función para calcular las medias y las desviaciones típicas.

		Args:
			datos (numpy.array): Array numpy con los datos.
			nominalAtributos (list): Lista con el tipo de las columnas (nominal o no; True o False)

		Returns:
			list: Lista de arrays en los cuales está la media y la desviación de cada columna.
		"""
		mediasDesv = []
		for i, col in enumerate(datos.transpose()):
			if not nominalAtributos[i]:
				mediasDesv.append(np.array([np.mean(col), np.std(col)]))
			else:
				mediasDesv.append(np.array([0, 0]))
		return mediasDesv

	def normalizarDatos(self, datos, nominalAtributos):
		"""Función para normalizar datos.

		Args:
			datos (numpy.array): Array numpy con los datos.
			nominalAtributos (list): Lista con el tipo de las columnas (nominal o no; True o False)

		Returns:
			numpy.array: Array numpy con los datos normalizados.
		"""
		mediasDevs = self.calcularMediasDesv(datos, nominalAtributos)
		datosNormalizados = np.zeros(shape=datos.shape)
		for numCol, col in enumerate(datos.transpose()):
			if nominalAtributos[numCol] or numCol == datos.shape[1]-1:
				datosNormalizados[:,numCol] = col
			else:
				for numRow in range(len(col)):
					datosNormalizados[numRow][numCol] = norm(mediasDevs[numCol][0], mediasDevs[numCol][1]).pdf(datos[numRow][numCol])
		return datosNormalizados

	def clasificaFila(self, fila):
		"""Dada una fila del test, buscamos las K clases más cercanas a la fila.

		Args:
			fila: Fila del test a comprobar

		Returns:
			Devuelve la clase adecuada dada la fila.
		"""
		distancias = [] # Lista con tuplas. [(distancia, clase), ...]
		# Calculamos las distancias
		for rowData in self.datosNormalizados:
			distancias.append((self.distanciaEuclidea(fila, rowData), rowData[-1]))

		return self.getClaseKVecinos(distancias)

	def getClaseKVecinos(self, distancias):
		"""Dadas unas distancias se devuelve la clase más frecuente.

		Args:
			distancias (list): Lista de tuplas que contiene (distancia, clase)

		Returns:
			float: Clase
		"""
		distanciasOrdenadas = sorted(distancias, key=lambda x: x[0])
		return  self.getClaseMasFrequente(distanciasOrdenadas[:self.K])

	def getClaseMasFrequente(self, distancias):
		clases = {}
		for _, c in distancias:
			if c not in clases:
				clases[c] = 1
			else:
				clases[c] += 1
		return sorted(clases.items(), key=lambda x: x[1], reverse=True)[0][0]

	def distanciaEuclidea(self, x, y, w=None):
		"""Función privada para calcular la distancia euclidea
		entre 2 vectores. También se calcula la distancia euclidea
		ponderada en caso de que se pasen las ponderaciones.

		Args:
			x (numpy.array): Vector.
			y (numpy.array): Vector.
			w (numpy.array): Vector con las ponderaciones.

		Returns:
			float: Distancia euclidea
		"""
		if w is None:
			return sum([(xi-yi)**2 for xi, yi in zip(x, y)])**(1/2)
		else:
			return sum([((xi-yi)*wi)**2 for xi, yi, wi in zip(x, y, w)])**(1/2)

	def distanciaManhattan(self, x, y):
		"""Función privada para calcular la distancia manhattan
		entre 2 vectores.

		Args:
			x (numpy.array): Vector.
			y (numpy.array): Vector.

		Returns:
			float: Distancia euclidea ponderada
		"""
		return [math.abs(xi-yi) for xi, yi in zip(x, y)]

	def distanciaChevychev(self, x, y):
		"""Función privada para calcular la distancia manhattan
		entre 2 vectores.

		Args:
			x (numpy.array): Vector.
			y (numpy.array): Vector.

		Returns:
			float: Distancia euclidea ponderada
		"""
		return [math.abs(xi-yi)**2 for xi, yi in zip(x, y)]

