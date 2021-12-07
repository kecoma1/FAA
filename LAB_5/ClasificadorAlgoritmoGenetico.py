from Clasificador import Clasificador
from random import randint
from collections import Counter
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
                {"regla": [randint(0, longitud-1) for longitud in longitudReglas],
                 "conlusion": randint(0, 1)}
                for _ in range(randint(1, maxReglas))
            ]
            self.fitness = -1

        def fitness(self, dataset):
            """Método para calcular el fitness de un individuo.

            Args:
                dataset (numpy.array): Matriz con los datos.

            Returns:
                float: Valor del fitness (entre [0-1]).
            """
            aciertos = sum([1 for dato in dataset if self.conclusion(dato)])
            return aciertos/dataset.shape[0]

        def conclusion(self, dato):
            """Método que obtiene la conclusión mayoritaria sobre
            un dato.

            Args:
                dato (numpy.array): Dato.

            Returns:
                int: Conclusión mayoritaria.
            """
            #conclusiones = [regla["conclusion"] if self.aplicaRegla(dato, regla) else 1 # TODO
            #                for regla in self.reglas]
            conclusiones = [regla["conclusion"] for regla in self.reglas 
                            if self.aplicaRegla(dato, regla)]
            if len(conclusiones) == 0:
                return 0 # TODO
            else:
                return Counter(conclusiones).most_common(1)[0][0]

        def aplicaRegla(self, dato, regla):
            """Dado un dato y una regla, aplicamos la regla sobre
            el dato y devolvemos la conclusión o TODO.

            Args:
                dato (numpy.array): Vector con los datos.
                regla (dict): Diccionario con la regla y la conclusión.

            Returns:
                bool: True si se cumplen todos los términos en la regla,
                False si no se cumple uno de ellos.
            """
            return all([True if regla["regla"][i] | (2**valor) else False 
                            for i, valor in enumerate(dato)])


    def __init__(self, poblacion, generaciones, maxReglas, cruce, mutacion, elitismo):
        self.poblacion = poblacion
        self.generaciones = generaciones
        self.maxReglas = maxReglas
        self.cruce = cruce
        self.mutacion = mutacion
        self.elitismo = elitismo
        self.individuos = []

    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        # La longitud máxima de cada regla corresponde a una cadena de bits con
        # todos sus valores a 1. Ejemplo: 3 posibles valores -> 111 = 7.
        # Lo que se hace es restar 1 al valor de la cadena con el bit más alto. 
        # Ejemplo 3 posibles valores -> 1000 = 8 = 2³ -> 8-1 = 7 = 0111 (el mismo número)
        longitudReglas = [(2**len(atr))-1 for atr in diccionario.values()]
        self.inicializarPoblacion(longitudReglas)

    def clasifica(self, datosTest, nominalAtributos, diccionario):
        return super().clasifica(datosTest, nominalAtributos, diccionario)

    def inicializarPoblacion(self, longitudReglas):
        """Método para inicializar la población.

        Args:
            longitudReglas (list): Lista con la longitud de cada regla
            en cada atributo.
        """
        for _ in range(self.poblacion):
            self.individuos.append(self.Individuo(self.maxReglas, longitudReglas))
