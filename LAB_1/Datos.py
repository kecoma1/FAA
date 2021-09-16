import pandas as pd
import numpy as np

class Datos:

    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero):
        self.nominalAtributos = []
        self.diccionario = {}
        self.datos = None
        
        
        df = pd.read_csv(nombreFichero)

        # Construimos la lista nominal atributos
        self.asignaNominalAtributos(df.dtypes)

        # Construimos el diccionario
        self.construyeDiccionario(df)
        
        # Construimos y traducimos el array de datos
        self.construyeDatos(df)

        
    # Devuelve el subconjunto de los datos cuyos �ndices se pasan como argumento
    def extraeDatos(self,idx):
        pass


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
        for item in df.iteritems():
            column_name = item[0]
            possible_values = list(df[column_name].unique()).copy()
            possible_values.sort()
            for i, value in enumerate(possible_values):
                if column_name not in self.diccionario:
                    self.diccionario[column_name] = {}
                self.diccionario[column_name][value] = i


    def construyeDatos(self, df):
        """Método que itera sobre todas las filas del dataset
        con el fin de traducir los datos a números.

        Args:
            df (Pandas Dataframe): Dataset con los datos originales.
        """
        self.datos = np.zeros(shape=df.shape)
        for i, item in enumerate(df.iterrows()):
            for j, col in enumerate(item[1].items()):
                self.datos[i][j] = self.diccionario[col[0]][col[1]]

    
    

