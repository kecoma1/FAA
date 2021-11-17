from operaciones import sigmoid
from Clasificador import Clasificador
from random import randint
import numpy as np


class ClasificadorRegresionLogistica(Clasificador):
	"""Clase que implementa el clasificador de regresión logística.
	Función para entrenar (actualización de pesos): 
		w' = w - constanteAprendizaje*(sigmoide(pesos*input)-clase)*input 
	
	Función para clasificar:
		sigmoide(pesos, input) > 0.5 -> 1
		sigmoide(pesos, input) < 0.5 -> 0
		sigmoide(pesos, input) == 0.5 -> Valor aleatorio entre 0 y 1.
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
				sigmoidJ =  sigmoid(np.dot(self.pesos, X))

				# Actualizamos los pesos
				self.pesos = self.pesos - (self.constanteAprendizaje*(sigmoidJ-clase))*X


	def clasifica(self, datostest, atributosDiscretos, _):
		res = []
		for row in datostest:
			X = row[:-1]
			prob = sigmoid(np.dot(self.pesos, X))
			if prob == 0.5:
				clase = randint(0, 1)
			else:
				clase = 1 if prob > 0.5 else 0
			res.append(clase)
		return res, []
