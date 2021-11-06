from Distancias import distanciaEuclidea
import numpy as np
import random


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


def kMeans(k, datos, maxIter=0):
	"""Función que calcula las K medias dado un dataset.
	Se asume que los datos que se envían no tienen incluida la columna "class"

	Args:
		k (int): Clusters a crear
		datos (numpy.array): Matriz numpy con los datos.
		maxIter (int): Argumento opcional con el que se establece un número máximo de
		iteraciones.

	Returns:
		dict: Diccionario con los clusters. La clave se refiere el número del cluster,
		no tiene nada que ver con la clase.
	"""
	# Lista con tuplas las cuales contienen: (Centroide, índice centroide en los datos)
	centroides = [(datos[centroide], cluster) for cluster, centroide in enumerate(eligeCentroides(k, len(datos)))]

	# Centroides anteriores, se usa esta variable para salir del bucle 
	prevCentroides = [(np.zeros(len(datos[0])), None) for _ in range(k)]

	# Para un número máximo de iteraciones
	i = 0
	if maxIter == 0:
		maxIter = -1	
		i = -2	

	while not comparaCentroides(prevCentroides, centroides) and i < maxIter:
		# Lista con los clusters a crear. La lista contiene los índices de cada
		# cluster y una lista en la que se meten los indices asignados a dicho cluster.
		clusters = {cluster: [] for _, cluster in centroides}

		for indiceDato, dato in enumerate(datos):
			# Para cada fila de los datos calculamos la distancia con cada centroide
			distanciasCentroides = [(distanciaEuclidea(dato, datosCentroide), cluster) for datosCentroide, cluster in centroides]
			
			# Ordenamos las distancias y obtenemos el cluster (centroide) más cercano.
			cluster = min(distanciasCentroides, key=lambda x: x[0])[1]
			clusters[cluster].append(indiceDato)

		# Recalculamos los centroides y reiniciamos los clusters
		prevCentroides = centroides
		centroides = calculaCentroides(clusters, datos)

		# En el caso de establecer un número máximo de iteraciones
		i = i+1 if maxIter != -1 else i

	return clusters, centroides


def confianzas(clusters, datos):
	"""Función para obtener la "confianza" de todos los clusters.

	Args:
		clusters (list): Lista con los clusters.
		datos (numpy.array): Dataset.

	Returns:
		dict: Diccionario con todas las confianzas. La "key" se refiere
		al cluster (la key dentro del diccionario de clusters), el "value"
		es una tupla con (confianza, clase mayoritaria)
	"""
	confianzas = {}
	numClases = len(np.unique(datos[:,-1]))

	# Analizamos la confianza de cada cluster
	for indiceCluster, cluster in clusters.items():
		confianza, clase = confianzaCluster(cluster, datos, numClases)
		confianzas[indiceCluster] = (confianza, clase)

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
		confianzas (dict): Diccionario con las confianzas de varios clusters.
		La "key" se refiere a la clase mayoritaria, y el "value" a la confianza de dicho cluster.

	Returns:
		Float: Media de las confianzas.
	"""
	avg = 0
	for confianza, _ in confianzas.values():
		avg += confianza

	return avg/len(confianzas)


def get_SK_clusters(labels, k):
    """Función para obtener un diccionario con los clusters
    dado el atributo "labels" del objeto SKLearn de KMeans.

    Args:
        labels: Array donde cada valor en este corresponde
        al cluster asociado de la fila en el dataset.
        k (int): Número de clusters.

    Returns:
        dict: Diccionario con los clusters.
    """
    clusters = { i: [] for i in range(k)}

    for index, cluster in enumerate(labels):
        clusters[cluster].append(index)

    return clusters
