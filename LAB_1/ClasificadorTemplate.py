from abc import ABCMeta, abstractmethod


class Clasificador:

	# Clase abstracta
	__metaclass__ = ABCMeta

	@abstractmethod
	def entrenamiento(self, datosTrain, nominalAtributos, diccionario):
		"""Metodos abstractos que se implementan en cada clasificador concreto

		Args:
			datosTrain: Matriz numpy con los datos de entrenamiento
			nominalAtributos: Array bool con la indicatriz de los atributos nominales
			diccionario: Array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
		"""
		pass

	@abstractmethod
	def clasifica(self, datosTest, nominalAtributos, diccionario):
		"""Esta funcion debe ser implementada en cada clasificador concreto. Devuelve un numpy array con las predicciones

		Args:
			datosTest: Matriz numpy con los datos de validación
			nominalAtributos: Array bool con la indicatriz de los atributos nominales
			diccionario: Array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
		"""
		pass

	def error(self, datos, pred):
		"""Obtiene el numero de aciertos y errores para calcular la tasa de fallo

		Args:
			datos: Matriz numpy con los datos de entrenamiento
			pred: Predicción
		"""
        # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
        pass

    
	def validacion(self, particionado, dataset, clasificador, seed=None):
		""" Realiza una clasificacion utilizando una estrategia de particionado determinada

		Args:
			particionado: Estrategia de particionado
			dataset: Dataset
			clasificador: Clasificador a usar
		"""

        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
		pass

##############################################################################


class ClasificadorNaiveBayes(Clasificador):

    # TODO: implementar
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        pass

    # TODO: implementar
    def clasifica(self, datostest, atributosDiscretos, diccionario):
        pass
