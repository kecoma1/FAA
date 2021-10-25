import numpy as np
import random
import math
from Datos import Datos


def distanciaEuclidea(x, y, w=None):
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


def distanciaManhattan(x, y):
	"""Función privada para calcular la distancia manhattan
	entre 2 vectores.

	Args:
		x (numpy.array): Vector.
		y (numpy.array): Vector.

	Returns:
		float: Distancia euclidea ponderada
	"""
	return [math.abs(xi-yi) for xi, yi in zip(x, y)]


def distanciaChevychev(x, y):
	"""Función privada para calcular la distancia manhattan
	entre 2 vectores.

	Args:
		x (numpy.array): Vector.
		y (numpy.array): Vector.

	Returns:
		float: Distancia euclidea ponderada
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


def centroDeMasas(puntosCluster):
	
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
	# Lista con tuplas las cuales contienen:
	# (Centroide, índice centroide en los datos)
	centroides = [(datos[centroide], cluster) for cluster, centroide in enumerate(eligeCentroides(k, len(datos)))]

	# Centroides anteriores, se usa esta variable para salir del bucle 
	prevCentroides = []

	# Lista con los clusters a crear. La lista contiene los índices de cada
	# cluster y una lista en la que se meten los indices asignados a dicho cluster.
	clusters = {cluster: [] for _, cluster in centroides}

	while prevCentroides != centroides:
		for indiceDato, dato in enumerate(datos):
			# Para cada fila de los datos calculamos la distancia con cada centroide
			distanciasCentroides = [(distanciaEuclidea(dato, datosCentroide), cluster) for datosCentroide, cluster in centroides]
			
			# Ordenamos las distancias y obtenemos el cluster (centroide)
			# más cercano.
			cluster = sorted(distanciasCentroides, key=lambda x: x[0])[0][1]
			clusters[cluster].append(indiceDato)

		# Recalculamos los centroides y reiniciamos los clusters
		prevCentroides = centroides
		centroides = calculaCentroides(clusters, datos)
		clusters = {centroide: [] for _, centroide in centroides}

	return clusters

datos = Datos("ConjuntosDatos/lentillas.data")
print(kMeans(5, datos.datos))
