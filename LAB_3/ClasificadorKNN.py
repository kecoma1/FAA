from Clasificador import Clasificador
import numpy as np
import Datos


class ClasificadorKNN(Clasificador):
	"""Clase donde se define el clasificardor K-NN.
	Este clasificador solo tiene un atributo, K, el cual es 
	necesario para la ejecución del algoritmo.

	Al clasificar se clasifica cada fila del test, con todas
	las filas del dataset. Para hacer esto es necesario guardar el 
	dataset. Esto se hace al ejecutar el método "entrenamiento",
	el cuál tiene un argumento opcional "norm" que por defecto está a False.
	Si se pusiese a True, se normalizarían los datos y se guardarían
	normalizados.
	"""

	def __init__(self, K, distanceFunc, norm=True):
		"""Constructor.

		Args:
			K (int): K a usar en el algoritmo KNN.
			distanceFunc (function): Función para calcular distancias.
			norm (bool, optional): Para normalizar datos o no. Por defecto a True.
		"""
		self.K = K
		self.distanceFunc = distanceFunc
		self.norm = norm

	def entrenamiento(self, datostrain, atributosDiscretos, _):
		# Calculamos las medias y las desviaciones
		self.mediasDesv = Datos.calcularMediasDesv(datostrain, atributosDiscretos) if self.norm else None

		# Calculamos la probabilidad de las clases
		self.calculaPClases(datostrain)

	def clasifica(self, datostest, atributosDiscretos, _):
		prediccionesClases = []
		testNormalizados =  Datos.normalizarDatos(datostest, atributosDiscretos, mediasDesv=self.mediasDesv)
		for rowTest in testNormalizados: # Por cada fila en el test
			prediccionesClases.append(self.clasificaFila(rowTest, datostest))
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

	def clasificaFila(self, fila, datos):
		"""Dada una fila del test, buscamos las K clases más cercanas a la fila.

		Args:
			fila: Fila del test a comprobar.
			datos: Datos a comprobar.

		Returns:
			Devuelve la clase adecuada dada la fila.
		"""
		distancias = [] # Lista con tuplas. [(distancia, clase), ...]
		# Calculamos las distancias
		for rowData in datos:
			distancias.append((self.distanceFunc(fila, rowData), rowData[-1]))

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
		return max(clases.items(), key=lambda x: x[1])[0]
