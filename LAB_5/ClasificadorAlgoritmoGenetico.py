from random import randint, random, choice
from Clasificador import Clasificador
from itertools import accumulate
from collections import Counter
from copy import copy
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

    @staticmethod
    def cruceInterReglas(A, B, _):
        """Método estático para cruzar 2 individuos 
        sustituyendo reglas enteras.

        Args:
            A (Individuo)
            B (Individuo)
            _ : Parametro no necesario

        Returns:
            tuple: Nuevo individuo A y nuevo individuo B
        """
        # Para calcular el punto de cruce tenemos en
        # cuenta el número de reglas

        puntoCruceA = 1 if len(A.reglas) == 1 else randint(1, len(A.reglas)-1)
        puntoCruceB = 1 if len(B.reglas) == 1 else randint(1, len(B.reglas)-1)

        newA = AlgoritmoGenetico.Individuo(0, 0, empty=True)
        newB = AlgoritmoGenetico.Individuo(0, 0, empty=True)

        # Cruzamos las reglas en A
        for i in range(0, puntoCruceA):
            newA.reglas.append({"regla": [], "conclusion": -1})
            newA.reglas[-1]["regla"] = A.reglas[i]["regla"]
            newA.reglas[-1]["conclusion"] = A.reglas[i]["conclusion"]

        for i in range(puntoCruceB, len(B.reglas)):
            newA.reglas.append({"regla": [], "conclusion": -1})
            newA.reglas[-1]["regla"] = B.reglas[i]["regla"]
            newA.reglas[-1]["conclusion"] = B.reglas[i]["conclusion"]

        # Cruzamos las reglas en B
        for i in range(0, puntoCruceB):
            newB.reglas.append({"regla": [], "conclusion": -1})
            newB.reglas[-1]["regla"] = B.reglas[i]["regla"]
            newB.reglas[-1]["conclusion"] = B.reglas[i]["conclusion"]

        for i in range(puntoCruceA, len(A.reglas)):
            newB.reglas.append({"regla": [], "conclusion": -1})
            newB.reglas[-1]["regla"] = A.reglas[i]["regla"]
            newB.reglas[-1]["conclusion"] = A.reglas[i]["conclusion"]
        return newA, newB

    @staticmethod
    def cruceIntraReglas(A, B, longitudReglas):
        """Método estático para cruzar 2 individuos
        sustituyendo partes de una regla.

        Args:
            A (Individuo)
            B (Individuo)
            longitudReglas (list): Lista con la longitud en
            bits de cada regla.

        Returns:
            tuple: Nuevo individuo A y nuevo individuo B.
        """
        # Para calcular el punto de cruce tenemos en
        # cuenta la longitud de todas las reglas juntas
        longitudReglasTotal = sum(longitudReglas)
        puntoCruce = randint(1, longitudReglasTotal-1)

        segundaMitad = (2**puntoCruce)-1
        primeraMitad = ((2**longitudReglasTotal)-1)-segundaMitad

        newA = AlgoritmoGenetico.Individuo(0, 0, empty=True)
        newB = AlgoritmoGenetico.Individuo(0, 0, empty=True)

        newA.reglas = A.reglas.copy()
        newB.reglas = B.reglas.copy()

        # Elegimos regla a intercambiar en cada individuo
        indiceReglaA = randint(0, len(A.reglas)-1)
        indiceReglaB = randint(0, len(B.reglas)-1)

        cadenaA = A.valorCadenaBitRegla(indiceReglaA, longitudReglas)
        primeraMitadA = cadenaA & primeraMitad
        segundaMitadA = cadenaA & segundaMitad

        cadenaB = B.valorCadenaBitRegla(indiceReglaB, longitudReglas)
        primeraMitadB = cadenaB & primeraMitad
        segundaMitadB = cadenaB & segundaMitad

        newReglaValueA = primeraMitadA | segundaMitadB
        newReglaValueB = primeraMitadB | segundaMitadA

        newA.setRegla(newReglaValueA, longitudReglas, indiceReglaA)
        newB.setRegla(newReglaValueB, longitudReglas, indiceReglaB)

        newA.reglas[indiceReglaA]["conclusion"] = B.reglas[indiceReglaB]["conclusion"]
        newB.reglas[indiceReglaB]["conclusion"] = A.reglas[indiceReglaA]["conclusion"]

        return newA, newB

    @staticmethod
    def mutacionEstandar(longitudReglas, individuos, pm):
        """Método estático para aplicar una mutación estandar
        a los individuos. Se elige una regla al azar y se cambia
        un bit al azar.

        Ejemplo:
        Si tenemos el número 5 y queremos cambiar el bit en la
        posición uno (101 | 010 -> 111) tenemos que tener en cuenta si
        este bit es 0 o 1. Si es 0 hay que cambiarlo a 1 hay que 
        usar un or con una cadena de bits con todo a 0s excepto el bit
        a modificar. En el caso de que queramos cambiar de 1 a 0
        (111 & 101 -> 101) Hay que crear una cadena identica a la original
        exceptuando el bit a cambiar y después aplicar la puerta lógica and.

        Args:
            longitudReglas (list): Lista con lo que mide cada regla.
            individuos (list): Individuos a los que aplicar la mutación.
            pm (float): Probabilidad de mutación.
        """
        for individuo in individuos:
            if random() <= pm:
                reglaACambiar = randint(0, len(individuo.reglas)-1)
                atributoACambiar = randint(0, len(individuo.reglas[reglaACambiar]["regla"]))
                bitACambiar = randint(0, longitudReglas[atributoACambiar]-1)

                # Cambiamos la conclusión
                if bitACambiar == len(individuo.reglas[reglaACambiar]["regla"]):
                    if individuo.reglas[reglaACambiar]["conclusion"] == 1:
                        individuo.reglas[reglaACambiar]["conclusion"] = 0
                    else:
                        individuo.reglas[reglaACambiar]["conclusion"] = 1
                else:
                    valorRegla = individuo.reglas[reglaACambiar]["regla"][atributoACambiar]
                    resultadoMutacion = valorRegla | (2**bitACambiar)
                    if resultadoMutacion == valorRegla:
                        resultadoMutacion = (valorRegla - (2**bitACambiar)) & valorRegla

                # Aplicamos la mutacion
                individuo.reglas[reglaACambiar]["regla"][atributoACambiar] = resultadoMutacion

    @staticmethod
    def mutacionReglas(longitudReglas, individuos, pm):
        """Método estático para mutar las reglas
        de los individuos en base a una probabilidad.

        Args:
            longitudReglas (list): Lista con la longitud de la cadena de
            bits de cada regla.
            individuos (list): Individuos a mutar.
            pm (float): Probabilidad a mutar una regla (0-1].
        """
        for individuo in individuos:
            for i, regla in enumerate(individuo.reglas):
                for n, valor in enumerate(regla["regla"]):
                    if random() <= pm:
                        bitACambiar = randint(0, longitudReglas[n])
                        resultadoMutacion = valor | (2**bitACambiar)
                        if resultadoMutacion == valor:
                            resultadoMutacion = (valor - (2**bitACambiar)) & valor

                        # Actualizamos la regla
                        individuo.reglas[i]["regla"][n] = resultadoMutacion
                # Cambiamos la conclusión
                if random() <= pm:
                    if regla["conclusion"] == 1:
                        regla["conclusion"] = 0
                    else:
                        regla["conclusion"] = 1

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

        def __init__(self, maxReglas, valorMaximoCadena, empty=False):
            """Constructor.

            Args:
                    maxReglas (int): Número máximo de reglas.
                    valorMaximoCadena (list): Lista con el valor máximo que se puede
                    alcanzar con la cadena de bits que representa los posibles valores
                    del atributo.
            """
            if not empty:
                self.reglas = [ # La llamada a randint es desde 1 hasta longitud-2 para
                                # que no haya reglas con todo a 1s
                    {"regla": [randint(1, longitud-2) if longitud-2 >= 1 else 1 for longitud in valorMaximoCadena],
                    "conclusion": randint(0, 1)}
                    for _ in range(randint(1, maxReglas-1))
                ]
            else:
                self.reglas = []
            self.fitnessValue = -1

        def fitness(self, dataset):
            """Método para calcular el fitness de un individuo.

            Args:
                dataset (numpy.array): Matriz con los datos.

            Returns:
                float: Valor del fitness (entre [0-1]).
            """
            aciertos = sum([1 for dato in dataset if self.conclusion(dato)])
            self.fitnessValue = aciertos/dataset.shape[0]
            return self.fitnessValue

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
            Ejemplo:
                Si tenemos la regla 110 (atr1=A o B o C) y tenemos el atributo C 001,
                no se cumple este término. Para comprobar esto hacemos 110 & 001 => 000
                Como el resultado no es igual al atributo (001) asumimos que no se
                cumple el término.

            Ejemplo2:
                Si tenemos la regla 110 (atr1=A o B o C) y tenemos el atributo A 100,
                se cumple este término. Para comprobar esto hacemos 110 & 100 => 100
                Como el resultado es igual al atributo (100) asumimos que se cumple el término.

            Args:
                dato (numpy.array): Vector con los datos.
                regla (dict): Diccionario con la regla y la conclusión.

            Returns:
                bool: True si se cumplen todos los términos en la regla,
                False si no se cumple uno de ellos.
            """
            for i, valor in enumerate(dato):
                if (regla["regla"][i] & (2**int(valor))) != 2**valor:
                    return False
            return True

        def valorCadenaBitRegla(self, indice, longitudReglas):
            """Función para calcular el valor de la cadena de bits
            que tiene una regla.

            Ejemplo: Si tenemos una regla con los valores 2, 5, 4, cuyas
            cadenas de bits son 10 101 100 (hay que tener en cuenta la longitud
            de cada regla porque el valor 5 también podría ser 0101, en este caso
            las longitudes son 2 3 3).
            Creamos una lista con los exponentes a aplicar a cada valor de la regla.
            exponentes = [6, 3, 0]
            Valor cadena 2*2⁶+5*2³+4*2⁰=172=10 101 100

            Args:
                indice (int): Índice de la regla a calcular.
                longitudReglas (list): Lista con la longitud de la cadena
                de bits de cada regla.

            Returns:
                int: Valor en base 10 de las reglas decimales.
            """
            longitudTotal = sum(longitudReglas)
            exponentes = []
            for longitud in longitudReglas:
                longitudTotal -= longitud
                exponentes.append(longitudTotal)
            return sum([v*(2**exp) for v, exp in zip(self.reglas[indice]["regla"], exponentes)])

        def setRegla(self, valorCadena, longitudReglas, indice):
            """Método para que dado un valor decimal de una cadena de bits
            se actualicen los valores en una regla especificada por un índice.
            Ejemplo: Si recibimos el número 172 = 10 101 100, y tenemos que guardar
            ese numero en 3 reglas con longitud 2, 3, 3, debemos hacer guardar
            2=10 5=101 4=100 en una lista con valores decimales.
            Para eso hacemos lo siguiente:
                Para guardar los 2 primeros bits tenemos que dividir la cadena
                entera entre 2⁶ (la cadena tiene 8 bits, sobran 6).
                10 101 100/00 100 000 = 10 - Guardamos este valor en la lista de las reglas.
                Ahora debemos actualizar el valor de la cadena para que en iteraciones
                futuras no se tengan en cuenta valores innecesarios.
                10 101 100 - 10 000 000 = 00 101 100 Este es el valor que usaremos 
                en la proxima iteración.
                Para obtener 10 000 000 lo que hacemos es usar una cadena con el tamaño maximo
                menos una cadena con los bits sobrantes a 0.
                11 111 111 - 00 111 111 = 11 000 000 Con este resultado hacemos & con
                el valor actual 10 101 100 & 11 000 000 = 10 000 000 ese valor es el que usamos
                para borrar. 

            Args:
                valorCadena (int): Valor decimal de la cadena.
                longitudReglas (list): Longitud de cada regla.
                indice (int): Regla a actualizar.
            """
            longitudTotal = sum(longitudReglas)
            todo1s = (2**longitudTotal)-1
            self.reglas[indice]["regla"] = []
            for longitud in longitudReglas:
                longitudTotal -= longitud
                self.reglas[indice]["regla"].append(int(valorCadena/(2**longitudTotal)))
                valorAEliminar = valorCadena & ( (todo1s) - ((2**longitudTotal)-1) )
                valorCadena -= valorAEliminar

    def __init__(self, poblacion, generaciones, maxReglas, cruce, mutacion, pm=0, elitismo=0, show=False):
        self.poblacion = poblacion
        self.generaciones = generaciones
        self.maxReglas = maxReglas
        self.cruce = cruce
        self.mutacion = mutacion
        self.pm = pm
        self.elitismo = elitismo
        self.individuos = []
        self.show = show

    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        # La longitud máxima de cada regla corresponde a una cadena de bits con
        # todos sus valores a 1. Ejemplo: 3 posibles valores -> 111 = 7.
        # Lo que se hace es restar 1 al valor de la cadena con el bit más alto. 
        # Ejemplo 3 posibles valores -> 1000 = 8 = 2³ -> 8-1 = 7 = 0111 == 111
        valorMaximoCadena = [(2**len(atr))-1 for atr in diccionario.values()]
        self.longitudReglas = [len(atr) for atr in diccionario.values()]
        self.inicializarPoblacion(valorMaximoCadena)

        self.evolucionPoblacion(datostrain)

    def clasifica(self, datosTest, nominalAtributos, diccionario):
        return super().clasifica(datosTest, nominalAtributos, diccionario)

    def inicializarPoblacion(self, valorMaximoCadena):
        """Método para inicializar la población.

        Args:
            valorMaximoCadena (list): Lista con el valor máximo que se puede
            alcanzar con la cadena de bits que representa los posibles valores
            del atributo.
        """
        for _ in range(self.poblacion):
            self.individuos.append(self.Individuo(self.maxReglas, valorMaximoCadena))

    def evolucionPoblacion(self, datostrain):
        individuosAMantener = int(self.poblacion * self.elitismo)
        mejoresIndividuos = [] # Lista con los mejores individuos de la anterior generación
        for iteracion in range(self.generaciones):
            print("Iteracion -", iteracion+1, end=" || ")

            # Calculamos el fitness de cada individuo
            fitnesses = [individuo.fitness(datostrain) 
                         if individuo.fitnessValue == -1 
                         else individuo.fitnessValue 
                         for individuo in self.individuos]

            if self.show: print("Fitness más alto:", max(fitnesses))

            # Guardamos los fitness más altos
            fitnessesIndizados = [(i, fitness) for i, fitness in enumerate(fitnesses)] 

            # Ordenamos por fitness los fitnesses
            fitnessesIndizadosOrdenados = sorted(fitnessesIndizados, key=lambda x: x[1], reverse=True)

            # Guardamos los indices de los mejores individuos
            mejoresIndividuosIndices = [i for i, _ in fitnessesIndizadosOrdenados[:individuosAMantener]]
            mejoresIndividuosIndices = sorted(mejoresIndividuosIndices, reverse=True) 

            # Guardamos los mejores individuos
            mejoresIndividuos = [copy(self.individuos[i]) for i in mejoresIndividuosIndices]

            # Borramos de la lista de individuos los mejores para no modificarlos
            for i in mejoresIndividuosIndices:
                del self.individuos[i]
                del fitnesses[i]

            # Calculamos la probabilidad de seleccion de cada individuo
            # en base al fitness total calculado
            totalFitness = sum(fitnesses) if sum(fitnesses) != 0 else 1
            probs = [fitness/totalFitness for fitness in fitnesses]

            # Mediante el método de la "ruleta" seleccionamos a los progenitores
            indicesProgenitores = self.ruleta(probs)

            # Cruzamos los progenitores para obtener los vastagos
            self.individuos = self.vastagos(indicesProgenitores).copy()

            # Mutamos la poblacion
            self.mutacion(self.longitudReglas, self.individuos, self.pm)

            # Volvemos a añadir a los mejores
            for individuo in mejoresIndividuos:
                self.individuos.append(individuo)

    def ruleta(self, probs):
        """Método para obtener los progenitores (sus índices en la lista de individuos)
        elegidos al azar mediante el método de la ruleta.

        Ejemplo: Lo primero que se hace es crear la ruleta en base a las probabilidades
        de cada individuo. Si tenemos como probabilidades 0.5, 0.2, 0.3 la ruleta sería
        la siguiente: 
            Zonas
            [0-0.5)     - Primer individuo
            [0.5-0.7)   - Segundo individuo
            [0.7-1)     - Tercer individuo
        Como se puede ver simplemente se suman las probabilidades de cada individuo.

        Ahora que tenemos la ruleta lanzamos la bola (llamada a random) tantas veces 
        como zonas haya en la ruleta, si obtenemos el valor 0.6, debemos guardar el 
        índice del segundo individuo, si obtenemos el valor 0.9, el del tercero.
        Para hacer esto cada vez que lanzamos la bola iteramos sobre las zonas y comprobamos
        si el valor donde cae la bola es menor que la zona que estamos comprobando (vamos en
        orden de menor a mayor con respecto al valor de la zona), en dicho caso el valor
        corresponde a la zona y guardamos el índice del individuo (ahora progenitor). 

        Args:
            probs (list): Lista con las probabilidades de cada individuo en base al fitness.

        Returns:
            list: Lista con los índices de los progenitores.
        """
        # Creamos la ruleta
        rule = list(accumulate(probs))

        # Dados los resultados vemos cuantas veces la bola
        # ha caido en cada lugar
        indices = []
        for _ in range(len(probs)):
            resultado = random()
            for i, zona in enumerate(rule):
                if resultado <= zona:
                    indices.append(i)
                    break
        if len(indices) == 0:
            indices = [randint(0, len(probs)-1) for _ in range(len(probs))]
        return indices

    def vastagos(self, indicesProgenitores):
        """Método para obtener los vástagos dados
        unos progenitores.

        Args:
            indicesProgenitores (list): Lista con los
            indices de los progenitores en la lista de
            individuos (self.individuos).

        Returns:
            list: Lista con los nuevos individuos.
        """
        newIndividuos = []
        pairs = self.parejas(indicesProgenitores)

        for A, B in pairs:
            newA, newB = self.cruce(self.individuos[A], self.individuos[B], self.longitudReglas)
            newIndividuos.append(newA)

            # Si tenemos un número impar de individuos 
            # en un cruce solo se incluira uno de los vastagos
            if len(newIndividuos) < self.poblacion:
                newIndividuos.append(newB)
        return newIndividuos

    def parejas(self, indicesProgenitores):
        """Método para obtener tuplas con las parejas de individuos
        a cruzar.

        Args:
            indicesProgenitores (list): Lista con los índices de los
            progenitores a cruzar.

        Returns:
            list: Lista de tuplas con las parejas.
        """
        parejas = []
        for i in range(0, len(indicesProgenitores), 2):
            A = indicesProgenitores[i]
            B = None
            if i+1 < len(indicesProgenitores):
                # Evitamos cruzar a un elemento consigo mismo buscando en el
                # resto de la lista
                if A == indicesProgenitores[i+1]:
                    for n in range(i+2, len(indicesProgenitores)):
                        posibleB = indicesProgenitores[n]
                        if posibleB != A:
                            B = posibleB
                            break

                    # Si en el resto de la lista no hemos encontrado una pareja,
                    # buscamos desde el principio de la lista
                    if B is None:
                        for n in range(0, i+1):
                            posibleB = indicesProgenitores[n]
                            if posibleB != A:
                                B = posibleB
                                break

                    # Si después de comprobar toda la lista no hay pareja,
                    # significa que todos los individuos son iguales, no
                    # nos queda otra que esperar una mutación.
                    if B is None:
                        B = indicesProgenitores[i+1]
                    parejas.append((A, B))
                else:
                    B = indicesProgenitores[i+1]
                    parejas.append((A, B))

            # Tenemos una lista de individuos impares, cruzaremos a este
            # ultimo individuo con uno al azar y después obtendremos solo
            # 1 vastago
            else:
                for i in range(len(indicesProgenitores)):
                    B = choice(indicesProgenitores)
                    if B != A:
                        parejas.append((A, B))
                        break
                if B is None:
                    parejas.append((A, A))

        # Si la lista de parejas no contiene todas las necesarias
        # lo que hacemos es repetir las parejas (se reproducen varias veces)
        i = 0
        while len(parejas) < (len(indicesProgenitores)/2):
            parejas.append(parejas[i])
            i = (i+1) % self.poblacion
        return parejas
