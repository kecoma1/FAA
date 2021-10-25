from numpy.core.fromnumeric import shape
from Clasificador import Clasificador
from scipy.stats import norm
from KMeans import distanciaEuclidea
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
		return prediccionesClases, []

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
			distancias.append((distanciaEuclidea(fila, rowData), rowData[-1]))

		return self.getClaseKVecinos(distancias)

	def getClaseKVecinos(self, distancias):
		"""Dadas unas distancias se devuelve la clase más frecuente.

		Args:
			distancias (list): Lista de tuplas que contiene (distancia, clase)

		Returns:
			float: Clase
		"""
		distanciasOrdenadas = sorted(distancias, key=lambda x: x[0])
		return  self.getClaseMasFrecuente(distanciasOrdenadas[:self.K])

	def getClaseMasFrecuente(self, distancias):
		"""Función que dadas unas distancias devuelve la
		clase más frecuente.

		Args:
			distancias (list): Lista de tuplas que contiene (distancia, clase).

		Returns:
			float: Clase.
		"""
		clases = {}
		for _, c in distancias:
			if c not in clases:
				clases[c] = 1
			else:
				clases[c] += 1
		# Devolvemos la calse con mayor frecuencia en K
		return max(clases.items(), key=lambda x: x[1])



