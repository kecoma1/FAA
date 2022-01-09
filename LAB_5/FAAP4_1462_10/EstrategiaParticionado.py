from abc import ABCMeta, abstractmethod
import random
import numpy as np

class Particion():
    """Clase que define una partición.
    """

    def __init__(self):
        """Constructor, solo se declaran los atributos.
        """
        self.indicesTrain=[]
        self.indicesTest=[]


class EstrategiaParticionado:
    """Clase abstracta donde se define la estrategia de particionado.
    """
    # Clase abstracta
    __metaclass__ = ABCMeta

    def __init__(self):
        """Constructor, solo se declaran los atributos.
        """
        self.particiones = []
    
    @abstractmethod
    def creaParticiones(self,datos,seed=None):
        """Método abstracto para crear particiones.

        Args:
            datos: Dataset.
            seed: Seed para generar aleatoriedad. Por defecto None.
        """
        pass


class ValidacionSimple(EstrategiaParticionado):
    """Clase que define una estrategia de particionado,
    en concreto, validación simple.
    """

    def __init__(self, proporcionTest, numeroEjecuciones):
        """Constructor.

        Args:
            proporcionTest: Porcentaje para la proporción de test de los datos.
            numeroEjecuciones: Número de ejecuciones.
        """
        super().__init__()
        self.proporcionTest = proporcionTest
        self.numeroEjecuciones = numeroEjecuciones
    
    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el n�mero de ejecuciones deseado
    # Devuelve una lista de particiones (clase Particion)
    def creaParticiones(self,datos,seed=None):
        random.seed(seed)
        self.particiones = []
        longitudDatos = np.shape(datos)[0]
        longitudTest = int((self.proporcionTest/100)*longitudDatos)
                
        lista_valores = [i for i in range(longitudDatos)]

        for _ in range(self.numeroEjecuciones):
            self.particiones.append(Particion())
            
            # Calculamos los indices
            random.shuffle(lista_valores)
            
            # Asignamos los indices
            self.particiones[-1].indicesTest = lista_valores[:longitudTest]
            self.particiones[-1].indicesTrain = lista_valores[longitudTest:]


class ValidacionCruzada(EstrategiaParticionado):
    """Clase que define una estrategia de particionado,
    en concreto, validación simple.
    """

    def __init__(self, numeroParticiones):
        """Construcor.

        Args:
            numeroParticiones (int): Número de particiones de la validación cruzada.
        """
        super().__init__()
        self.numeroParticiones = numeroParticiones

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    def creaParticiones(self,datos,seed=None):
        random.seed(seed)
        self.particiones = []
        longitudDatos = np.shape(datos)[0]
        longitudPorcion = int(longitudDatos/self.numeroParticiones)
        
        lista_valores = [i for i in range(longitudDatos)]
        random.shuffle(lista_valores)
            
        for i in range(self.numeroParticiones):
            self.particiones.append(Particion())

            # Calculamos los indices del test
            fromTest = i*longitudPorcion
            toTest = fromTest + longitudPorcion

            # Asignamos los indices
            self.particiones[-1].indicesTest = lista_valores[fromTest:toTest]
            self.particiones[-1].indicesTrain = [i for i in lista_valores if i not in self.particiones[-1].indicesTest]