from Clasificador import Clasificador
from random import randint
import numpy as np
import Datos

class AlgoritmoGenetico(Clasificador):
	"""Clase donde se define un algoritmo genético.
	Las reglas son conjuntos de bits, que se pueden representar mediante números,
	para operar estos números (como si fuesen bits) solo es necesario usar bit operators.

	Si tenemos el atributo X, con los valores A, B, C  (0, 1, 2 respectivamente en el 
	diccionario de datos) podemos representar este atributo como una cadena de bits:
	100=0 (A, no B, no C) ó 101=5 (A ó C, no B). La longitud de la regla sería el valor 7,
	ya que este es el valor más alto que se puede alcanzar mediante 3 bits.

	Para comparar un ejemplo con una regla hay que hacer lo siguiente:
	Si el ejemplo tiene el atributo X con valor C, no hay que pasarle un 2, hay que pasar
	su posicion en una cadena binaria. Para eso lo único que hay que hacer es 2²=4 -> 100.
	En caso de que fuese B pues 2¹=2-> 010
	"""

	class Individuo:
		"""Definición de un individuo en la población.
		Básicamente esta formado por un conjunto de reglas,
		la conclusión y el fitness.

		Las reglas se guardan en orden en una lista. Si tenemos
		atributo 1, 2, 3, la lista tendra las reglas de cada atributo
		en el respectivo orden.

		La lista de reglas es algo así:
		[
			{"regla": [7, 2, ...], "conclusion": 1}
			{"regla": [7, 3, ...], "conclusion": 0}
			{"regla": [1, 3, ...], "conclusion": 1}
		]
		"""

		def __init__(self, maxReglas, longitudReglas):
			"""Constructor.

			Args:
				maxReglas (int): Número máximo de reglas.
				longitudReglas (list): Lista con la longitud máxima de
				cada regla.
			"""
			self.reglas = [
						  	{"regla": np.zeros(len(longitudReglas)), 
							 "conlusion": randint(0, 1)} 
						  	for _ in range(randint(1, maxReglas))
						  ]
			for regla in self.reglas:
				for longitud in longitudReglas:
					# No se admiten todo 0's o 1's por eso desde 1 hasta longitud-1
					regla["regla"] = randint(1, longitud-1) 
			self.fitness = -1

		def fitness(dataset, diccionario):
			"""Dado un dataset, calculamos el fitness del individuo.

			Args:
				dataset (numpy.array): Dataset.
			"""
			pass


	def __init__(self, poblacion, generaciones, maxReglas, cruce, mutacion, elitismo):
		self.poblacion = poblacion
		self.generaciones = generaciones
		self.maxReglas = maxReglas
		self.cruce = cruce
		self.mutacion = mutacion
		self.elitismo = elitismo
		self.individuos = []

	def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
		longitudReglas = [len(atr) for atr in diccionario.Items()]
		self.inicializarPoblacion(longitudReglas)
		pass

	def clasifica(self, datosTest, nominalAtributos, diccionario):
		return super().clasifica(datosTest, nominalAtributos, diccionario)

	def inicializarPoblacion(self, longitudReglas):
		for _ in range(self.poblacion):
			self.individuos.append(self.Individuo(self.maxReglas, longitudReglas))