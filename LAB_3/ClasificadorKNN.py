from Clasificador import Clasificador
from scipy.stats import norm
import numpy as np


class ClasificadorKNN(Clasificador):

	def __init__(self, K):
		self.K = K

	def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
		pass

	def clasifica(self, datostest, atributosDiscretos, diccionario):
		pass

	def calcularMediasDesv(self, datos, nominalAtributos):
		mediasDesv = []
		for i, col in enumerate(datos.transpose()):
			if not nominalAtributos[i]:
				mediasDesv.append(np.array([np.mean(col), np.std(col)]))
			else:
				mediasDesv.append(np.array([0, 0]))
		return mediasDesv

	def normalizarDatos(self, datos, nominalAtributos):
		mediasDevs = self.calcularMediasDesv(datos, nominalAtributos)
		for i, row in enumerate(datos):
			for j in range(len(row)):
				if nominalAtributos[j]:
					continue
				datos[i][j] = norm(mediasDevs[j][0], mediasDevs[j][1]).pdf(datos[i][j])