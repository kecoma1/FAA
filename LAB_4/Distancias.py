import math


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