from abc import ABCMeta,abstractmethod
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
        self.clasificador = None
        self.datos = None
    
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
        particiones = []
        longitudDatos = np.shape(datos)[0]
        
        # Permutamos los datos (solo las filas)
        np.random.shuffle(datos)
        random.seed(seed)
        
        for i in range(self.numeroEjecuciones):
            particiones.append(Particion())
            
            # Calculamos los indices
            fromTest = int(random.randrange(0, longitudDatos))
            longitudTest = int((self.proporcionTest/100)*longitudDatos)
            toTest = fromTest+longitudTest if fromTest+longitudTest <= longitudDatos else longitudTest-(longitudDatos-fromTest)
            
            # Asignamos los indices
            particiones[-1].indicesTest = [fromTest, toTest]
            particiones[-1].indicesTrain = [toTest, fromTest]
        
        return particiones


class ValidacionCruzada(EstrategiaParticionado):
    
    def __init__(self, numeroParticiones):
        super().__init__()
        self.numeroParticiones = numeroParticiones
        
  
    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self,datos,seed=None):   
        random.seed(seed)
        pass
    
