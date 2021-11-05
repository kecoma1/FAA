from numpy.core.fromnumeric import shape
from Datos import Datos
from scipy.stats import norm
import numpy as np
from ClasificadorKNN import ClasificadorKNN
from EstrategiaParticionado import ValidacionCruzada, ValidacionSimple
import KMeans
import MatrizConfusion as MC

#datos1 = Datos("ConjuntosDatosP2Knn/wdbc.data")
#datos1 = Datos("ConjuntosDatos/tic-tac-toe.data")
#datos1 = Datos("ConjuntosDatos/german.data")

knn = ClasificadorKNN(25, norm=True)
datos1 = Datos("ConjuntosDatosP2Knn/pima-indians-diabetes.data")
print("KNN | norm=True | pima-indians-diabetes:", knn.validacion(ValidacionCruzada(10), datos1, knn))

knn = ClasificadorKNN(25, norm=False)
datos1 = Datos("ConjuntosDatosP2Knn/pima-indians-diabetes.data")
print("KNN | norm=False | pima-indians-diabetes:", knn.validacion(ValidacionCruzada(10), datos1, knn))


knn = ClasificadorKNN(25, norm=True)
datos1 = Datos("ConjuntosDatosP2Knn/wdbc.data")
print("KNN | norm=True | wbc:", knn.validacion(ValidacionCruzada(10), datos1, knn))

knn = ClasificadorKNN(25, norm=False)
datos1 = Datos("ConjuntosDatosP2Knn/wdbc.data")
print("KNN | norm=False | wbc:", knn.validacion(ValidacionCruzada(10), datos1, knn), end="\n\n")



datos1 = Datos("ConjuntosDatosP2KMeans/nums.csv")
clusters = KMeans.kMeans(10, datos1.datos)
confianzas = KMeans.confianzas(clusters, datos1.datos)
media = KMeans.confianzaMedia(confianzas)
for indiceCluster, cluster in clusters.items():
	vp, vn, fp, fn = MC.matrizConfusionCluster(cluster, datos1.datos, confianzas[indiceCluster][1], resInProb=True)
	print("Cluster", indiceCluster, "- Clase mayoritaria:", confianzas[indiceCluster][1])
	print("Exactitud:", MC.exactitud(vp, vn, fp, fn))
	print("Precision:", MC.precision(vp, fp))
	print("Sensibilidad:", MC.sensibilidad(vp, fn))
	print("Especificidad:", MC.especificidad(vn, fp), end="\n=================================\n")

#print(datos1.datos[:,1])
#normalizarDatos(datos1.datos, datos1.nominalAtributos)
#print(datos1.datos[:,1])
