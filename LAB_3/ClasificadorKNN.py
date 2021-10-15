from numpy.core.fromnumeric import shape
from Clasificador import Clasificador
from scipy.stats import norm
import numpy as np
import math


class ClasificadorKNN(Clasificador):

	def __init__(self, K):
		self.K = K

	def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
		pass

	def clasifica(self, datostest, atributosDiscretos, diccionario):
		pass

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
		for i, row in enumerate(datos):
			for j in range(len(row)):
				if nominalAtributos[j]:
					continue
				datosNormalizados[i][j] = norm(mediasDevs[j][0], mediasDevs[j][1]).pdf(datos[i][j])
		return datosNormalizados

	def __distanciaEuclidea(self, x, y, w=None):
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

	def __distanciaManhattan(self, x, y):
		"""Función privada para calcular la distancia manhattan
		entre 2 vectores.

		Args:
			x (numpy.array): Vector.
			y (numpy.array): Vector.

		Returns:
			float: Distancia euclidea ponderada
		"""
		return [math.abs(xi-yi) for xi, yi in zip(x, y)]

	def __distanciaChevychev(self, x, y):
		"""Función privada para calcular la distancia manhattan
		entre 2 vectores.

		Args:
			x (numpy.array): Vector.
			y (numpy.array): Vector.

		Returns:
			float: Distancia euclidea ponderada
		"""
		return [math.abs(xi-yi)**2 for xi, yi in zip(x, y)]

