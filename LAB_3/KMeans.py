import numpy as np
import random
import math
from Datos import Datos


def distanciaEuclidea(x, y, w=None):
	"""Función para calcular la distancia euclidea
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


def distanciaManhattan(x, y):
	"""Función para calcular la distancia manhattan
	entre 2 vectores.

	Args:
		x (numpy.array): Vector.
		y (numpy.array): Vector.

	Returns:
		float: Distancia Manhattan
	"""
	return [math.abs(xi-yi) for xi, yi in zip(x, y)]


def distanciaChevychev(x, y):
	"""Función para calcular la distancia manhattan
	entre 2 vectores.

	Args:
		x (numpy.array): Vector.
		y (numpy.array): Vector.

	Returns:
		float: Distancia Chevychev
	"""
	return [math.abs(xi-yi)**2 for xi, yi in zip(x, y)]


def eligeCentroides(k, longitudDatos):
	"""Función para elegir los centroides.
	Se devuelven los índices dentro del array.

	Args:
		k (int): K centroides a elegir.
		longitudDatos (int): Número de ejemplos en el dataset.

	Returns:
		list: Lista con los centroides.
	"""
	randoms = []
	while len(randoms) < k:
		n = random.randrange(0, longitudDatos-1)
		if n not in randoms:
			randoms.append(n)
	return randoms


def comparaCentroides(l1, l2):
	"""Función que compara centroides.

	Args:
		l1 (list): Lista con centroides.
		l2 (list): Lista con centroides.

	Returns:
		bool
	"""
	for c1, c2 in zip(l1, l2):
		for i1, i2 in zip(c1[0], c2[0]):
			if i1 != i2:
				return False
	return True


def centroDeMasas(puntosCluster):
	"""Función para calcular el centro de masas dados varios puntos en un cluster.
	***Si hay columnas nominales el centro de masas se calcula de la misma
	forma que con las columnas que no son nominales.
	Por ejemplo, si tenemos una columna nominal con los valores AZUL = 0 y ROJO = 1,
	es posible obtener el valor 0.5 en  el centro de masas aunque este valor no exista.

	Args:
		puntosCluster (list): Lista con los puntos del cluster.

	Returns:
		numpy.array: Array con los valores del centro de masas.
	"""
	centro = np.zeros(puntosCluster.shape[1])
	for i, col in enumerate(puntosCluster.transpose()):
		centro[i] = sum(col)/len(col)
	return centro


def calculaCentroides(clusters, datos):
	"""Función para calcular los centroides de unos
	clusters calculando el centro de masas.

	Args:
		clusters (dict): Diccionario con los clusters.
		datos (numpy.array): Array con los datos.

	Returns:
		list: Lista con los nuevos centroides.
	"""
	return [(centroDeMasas(datos[cluster]), i) for i, cluster in enumerate(clusters.values())]


def kMeans(k, datos):
	"""Función que calcula las K medias dado un dataset.
	Se asume que los datos que se envían no tienen incluida la columna "class"

	Args:
		k (int): Clusters a crear
		datos (numpy.array): Matriz numpy con los datos.

	Returns:
		list: Lista con los clusters.
	"""
	# Lista con tuplas las cuales contienen: (Centroide, índice centroide en los datos)
	centroides = [(datos[centroide], cluster) for cluster, centroide in enumerate(eligeCentroides(k, len(datos)))]

	# Centroides anteriores, se usa esta variable para salir del bucle 
	prevCentroides = [(np.zeros(len(datos[0])), None) for _ in range(k)]

	while not comparaCentroides(prevCentroides, centroides):
		# Lista con los clusters a crear. La lista contiene los índices de cada
		# cluster y una lista en la que se meten los indices asignados a dicho cluster.
		clusters = {cluster: [] for _, cluster in centroides}

		for indiceDato, dato in enumerate(datos):
			# Para cada fila de los datos calculamos la distancia con cada centroide
			distanciasCentroides = [(distanciaEuclidea(dato, datosCentroide), cluster) for datosCentroide, cluster in centroides]
			
			# Ordenamos las distancias y obtenemos el cluster (centroide) más cercano.
			cluster = sorted(distanciasCentroides, key=lambda x: x[0])[0][1]
			clusters[cluster].append(indiceDato)

		# Recalculamos los centroides y reiniciamos los clusters
		prevCentroides = centroides
		centroides = calculaCentroides(clusters, datos)

	return clusters


def confianzas(clusters, datos):
	"""Función para obtener la "confianza" de todos los clusters.

	Args:
		clusters (list): Lista con los clusters.
		datos (numpy.array): Dataset.

	Returns:
		list: Lista con todas las confianzas
	"""
	confianzas = []
	numClases = len(np.unique(datos[:,-1]))

	# Analizamos la confianza de cada cluster
	for cluster in clusters.values():
		confianzas.append(confianzaCluster(cluster, datos, numClases))

	return confianzas


def confianzaCluster(cluster, datos, numClases):
	"""Función para calcular la confianza en específico de un cluster.
	Para calcular esta confianza se obtiene la clase mayoritaria en
	el cluster para posteriormente calcular el porcentaje de está clase
	dentro del cluster.

	Args:
		cluster (list): Lista con las filas de los datos agrupadas en el cluster.
		datos (numpy.array): Dataset.
		numClases (int): Cantidad de clases en el dataset.

	Returns:
		(float, int): Porcentaje de la clase mayoritaria en el cluster y la clase mayoritaria.
	"""
	clasesFreq = {i: 0 for i in range(numClases)}

	# Analizamos la frecuencia de las clases en el cluster
	for dato in cluster:
		claseDato = datos[dato][-1]
		clasesFreq[int(claseDato)] += 1

	claseMayoritaria, freqMayoritaria = max(clasesFreq.items(), key=lambda x: x[1])
	return (freqMayoritaria/sum(clasesFreq.values()), claseMayoritaria)


def confianzaMedia(confianzas):
	"""Función para obtener la media de todas las confianzas.

	Args:
		confianzas (list): Lista con las confianzas de varios clusters.

	Returns:
		Float: Media de las confianzas.
	"""
	avg = 0
	for confianza in confianzas:
		avg += confianza[0]

	return avg/len(confianzas)


def matrizConfusionCluster(cluster, datos, clase, resInProb=False):
	"""Función que dado un cluster, devuelve los datos de una matríz de confusión.

	Args:
		cluster (tuple): Cluster a analizar.
		datos (numpy.array): Dataset
		clase (int): Clase mayoritaria en el cluster
		resInProb (bool): Variable opcional para obtener los valores 
		con valores entre 0-1

	Returns:
		tuple: Verdaderos positivos, verdaderos negativos, falsos positivos, falsos negativos.
	"""
	lenCluster = 1
	lenDatos = 1
	vp = verdaderosPositivos(cluster, datos, clase)
	vn = verdaderosNegativos(cluster, datos, clase)
	fp = falsosPositivos(cluster, datos, clase)
	fn = falsosNegativos(cluster, datos, clase)

	if resInProb:
		lenCluster = len(cluster)
		lenDatos = len(datos)

	# Si resInProb es False, lenDatos=1 y lenCluster=1.
	return vp/lenCluster, vn/lenDatos, fp/lenCluster, fn/lenDatos


def verdaderosPositivos(cluster, datos, clase):
	"""Función para calcular los verdaderos positivos.
	Obtenemos el número de datos dentro del cluster
	que pertenece a la clase mayoritaria.

	Args:
		cluster (list): Lista con los indices de los datos en el cluster.
		datos (numpy.array): Dataset.
		clase (int): Clase mayoritaria del cluster.

	Returns:
		int: Número de verdaderos positivos en el cluster.
	"""
	vp = 0
	for dato in cluster:
		if int(datos[dato][-1]) == clase:
			vp += 1
	return vp 


def verdaderosNegativos(cluster, datos, clase):
	"""Función para calcular los verdaderos negativos.
	Obtenemos el número de datos en el dataset que:
		1. No están en el cluster
		2. La clase es distinta

	Args:
		cluster (list): Lista con los indices de los datos en el cluster.
		datos (numpy.array): Dataset.
		clase (int): Clase mayoritaria del cluster.

	Returns:
		int: Número de verdaderos negativos en el cluster.
	"""
	vn = 0
	for i, fila in enumerate(datos):
		if fila[-1] != clase and i not in cluster:
			vn += 1
	return vn


def falsosPositivos(cluster, datos, clase):
	"""Función para calcular los falsos positivos.
	Obtenemos el número de datos en el cluster que
	no son de la clase mayoritaria.

	Args:
		cluster (list): Lista con los indices de los datos en el cluster.
		datos (numpy.array): Dataset.
		clase (int): Clase mayoritaria del cluster.

	Returns:
		int: Número de falsos positivos en el cluster.
	"""
	fp = 0
	for dato in cluster:
		if int(datos[dato][-1]) != clase:
			fp += 1
	return fp


def falsosNegativos(cluster, datos, clase):
	"""Función para calcular los falsos negativos.
	Obtenemos el número de datos en el dataset que:
		1. No están en el cluster.
		2. Son de la clase mayoritaria del cluster.

	Args:
		cluster (list): Lista con los indices de los datos en el cluster.
		datos (numpy.array): Dataset.
		clase (int): Clase mayoritaria del cluster.

	Returns:
		int: Número de falsos negativos en el cluster.
	"""
	fn = 0
	for i, fila in enumerate(datos):
		if fila[-1] == clase and i not in cluster:
			fn += 1
	return fn