import pandas as pd
import numpy as np

class Datos:

    nominalAtributos = []
    diccionario = {}

    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero):
        df = pd.read_csv(nombreFichero)

        # Obtenemos el tipo de cada columna
        tipos = df.dtypes

        # Construimos la lista nominal atributos
        self.asignaNominalAtributos(tipos)

        # Construimos el diccionario



        
    # Devuelve el subconjunto de los datos cuyos �ndices se pasan como argumento
    def extraeDatos(self,idx):
        pass

    def asignaNominalAtributos(self, tipos):
        """Itera sobre los tipos de atributos y asigna:
            True: Si la columna es nominal (object)
            False: Si la columna no es nominal 

        Args:
            line: Tipos obtenidos del dataframe de pandas
        """
        for tipo in tipos:
            self.nominalAtributos.append(True if tipo == object else False)
    
    def construyeDiccionario(self, df):
        """Método que itera sobre los datos y construye
        un diccionario de diccionario en el que se muestra
        el valor que puede tener cada atributo. Ejemplo:
        {
            "Attr1": {"x": 1, "y": 2}
            "Attr2": {"x": 1, "y": 2}
            ...
        }

        Args:
            df (Pandas Dataframe): Dataframe pandas con todos los datos
        """
        # Iteramos sobre todos los datos para comprobar todos los posibles valores
        for line in df.iterrows():
            keys = line[1].keys()
            values = line[1].values()
            for i, key in enumerate(keys):
                self.diccionario[key] = values[i] if values[i] not in self.diccionario[key]
                

