import math


def sigmoid(x):
	"""Función que calcula la sigmoide dado un valor X.

	Args:
		x (float): Valor X.

	Returns:
		float: Sigmoide.
	"""
	if x > 709:
		return 1
	elif x < -709:
		return 0
	return 1 / (1 + math.exp(-x))


def productoEscalar(a, b):
	"""Función que devuelve el producto escalar de 2 vectores.

	Args:
		a (np.array): Vector 1.
		b (np.array): Vector 2.

	Returns:
		float: Producto escalar.
	"""
	return sum([ai*bi for ai, bi in zip(a, b)])