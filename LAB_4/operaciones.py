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
