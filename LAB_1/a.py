from Datos import Datos
import pandas as pd
import numpy as np
import random
import EstrategiaParticionado

datos = np.random.random((10, 2)) 

#vc = EstrategiaParticionado.ValidacionCruzada(3)
#particiones = vc.creaParticiones(datos)

vc = EstrategiaParticionado.ValidacionSimple(75, 3)
vc.creaParticiones(datos)
      
for i in range(3):
    print()
    print("Test:", vc.particiones[i].indicesTest)
    print("Train:", vc.particiones[i].indicesTrain)

datos = Datos("ConjuntosDatos/tic-tac-toe.data")
indata = datos.extraeDatos([0, 1, 2, 3, 4])

for i in range(5):
    print(datos.datos[i])
    print(indata[i])
    print(indata[i] == datos.datos[i])
    print()