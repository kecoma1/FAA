from random import shuffle
import pandas as pd
import numpy as np

class Datos:

    def __init__(self, nombreFichero):
        """Constructor de la clase datos.

        Args:
            nombreFichero (str): Nombre del fichero con los datos.
        """
        self.nominalAtributos = []
        self.diccionario = {}
        self.datos = None
        
        # Dataframe pandas
        df = pd.read_csv(nombreFichero)

        # Construimos la lista nominal atributos
        self.asignaNominalAtributos(df.dtypes)

        # Construimos el diccionario
        self.construyeDiccionario(df)
        
        # Construimos y traducimos el array de datos
        self.construyeDatos(df)

        
    # Devuelve el subconjunto de los datos cuyos �ndices se pasan como argumento
    def extraeDatos(self,idx):
        datos = np.zeros(shape=(len(idx), self.datos.shape[1]))

        for i, index in enumerate(idx):
            datos[i] = self.datos[index]
        
        return datos


    def asignaNominalAtributos(self, tipos):
        """Itera sobre los tipos de atributos y asigna:
            True: Si la columna es nominal (object).
            False: Si la columna no es nominal.

        Args:
            line: Tipos obtenidos del dataframe de pandas.
        """
        for tipo in tipos:
            self.nominalAtributos.append(True if tipo == object else False)


    def construyeDiccionario(self, df):
        """Método que itera sobre los datos y construye
        un diccionario de diccionarios en el que se muestra
        el valor que puede tener cada atributo. Ejemplo:
        {
            "Attr1": {"x": 1, "y": 2} -- Orden alfabético
            "Attr2": {"x": 1, "y": 2}
            ...
        }
        
        Args:
            df (Pandas Dataframe): Dataframe pandas con todos los datos.
        """
        for i, item in enumerate(df.iteritems()):
            columnName = item[0]
            possibleValues = list(df[columnName].unique()).copy()
            possibleValues.sort()
            for n, value in enumerate(possibleValues):
                if not self.nominalAtributos[i]:
                    continue
                if columnName not in self.diccionario:
                    self.diccionario[columnName] = {}
                self.diccionario[columnName][value] = n


    def construyeDatos(self, df):
        """Método que itera sobre todas las filas del dataset
        con el fin de traducir los datos a números.

        Args:
            df (Pandas Dataframe): Dataset con los datos originales.
        """
        self.datos = np.zeros(shape=df.shape)
        for i, item in enumerate(df.iterrows()):
            for j, col in enumerate(item[1].items()):
                if self.nominalAtributos[j]:
                    self.datos[i][j] = self.diccionario[col[0]][col[1]]
                else:
                    self.datos[i][j] = col[1]
