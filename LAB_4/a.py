from ClasificadorRegresionLogistica import ClasificadorRegresionLogistica
from Datos import Datos
import numpy as np
import math

wdbc = Datos("ConjuntosDatos/wdbc.data")

#crl = ClasificadorRegresionLogistica(1, 10)
#crl.entrenamiento(wdbc.datos, wdbc.nominalAtributos, wdbc.diccionario)


for i in range(-600, -1000, -1):
	print(i)
	math.exp(i)