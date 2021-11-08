from Clasificador import Clasificador
from operaciones import productoEscalar, sigmoid
import numpy as np
import Datos


class ClasificadorRegresionLogistica(Clasificador):
	"""
	w' = w - constanteAprendizaje*(sigmoide(pesos*input)-clase)*input 
	"""

	def __init__(self, constanteAprendizaje, epocas):
		self.constanteAprendizaje = constanteAprendizaje
		self.epocas = epocas
		self.pesos = None

	def entrenamiento(self, datostrain, atributosDiscretos, _):
		# Inicializamos los pesos con valores [-0.5, 0.5]
		self.pesos = np.random.uniform(low=-0.5, high=0.5, size=(len(atributosDiscretos)-1,))

		for _ in range(self.epocas): # En cada época
			for row in datostrain: # Recorremos todos los ejemplos.
				X = row[:-1]
				clase = row[-1]
				sigmoidJ =  sigmoid(productoEscalar(self.pesos, X))

				# Actualizamos los pesos
				self.pesos = self.pesos - (self.constanteAprendizaje*(sigmoidJ-clase))*X


	def clasifica(self, datostest, atributosDiscretos, _):
		res = []
		for row in datostest:
			X = row[:-1]
			prob = sigmoid(productoEscalar(self.pesos, X))

			res.append(row[-1] if prob > 0.5 else )

