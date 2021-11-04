def matrizConfusionCluster(cluster, datos, clase, resInProb=False):
	"""Función que dado un cluster, devuelve los datos de una matríz de confusión.

	Args:
		cluster (tuple): Cluster a analizar.
		datos (numpy.array): Dataset
		clase (int): Clase mayoritaria en el cluster
		resInProb (bool): "Result In Probability". Variable opcional para obtener los valores 
		con valores entre 0-1

	Returns:
		tuple: Verdaderos positivos, verdaderos negativos, falsos positivos, falsos negativos.
	"""
	p = 1
	n = 1
	vp = verdaderosPositivos(cluster, datos, clase)
	vn = verdaderosNegativos(cluster, datos, clase)
	fp = falsosPositivos(cluster, datos, clase)
	fn = falsosNegativos(cluster, datos, clase)

	if resInProb:
		p = positivos(datos, clase)
		n = negativos(datos, clase)

	# Si resInProb es False, lenDatos=1 y lenCluster=1.
	return vp/p, vn/n, fp/p, fn/n


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


def positivos(datos, clase):
	"""Función para obtener el número de negativos

	Args:
		datos (numpy.array): Dataset.
		clase (int): Clase mayoritaria del cluster.

	Returns:
		int: Número de falsos negativos en el cluster.
	"""
	p = 0
	for fila in datos:
		if fila[-1] == clase:
			p += 1
	return p


def negativos(datos, clase):
	"""Función para obtener el número de negativos

	Args:
		datos (numpy.array): Dataset.
		clase (int): Clase mayoritaria del cluster.

	Returns:
		int: Número de falsos negativos en el cluster.
	"""
	n = 0
	for fila in datos:
		if fila[-1] != clase:
			n += 1
	return n