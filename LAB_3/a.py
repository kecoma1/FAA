from numpy.core.fromnumeric import shape
from Datos import Datos
from scipy.stats import norm
import numpy as np
from ClasificadorKNN import ClasificadorKNN
from Distancias import distanciaEuclidea
from EstrategiaParticionado import ValidacionCruzada, ValidacionSimple
import ClusteringKMeans
import MatrizConfusion as MC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans as SKKMeans
from sklearn.model_selection import cross_val_score, train_test_split
import utils

nums = Datos("ConjuntosDatosP2KMeans/nums.csv")
wdbc = Datos("ConjuntosDatosP2Knn/wdbc.data")
pima = Datos("ConjuntosDatosP2Knn/pima-indians-diabetes.data")

matrices_propia_10, _ = utils.test_KMeans(10, nums)
matrices_SK_10, centroides = utils.test_KMeans_SK(10, nums)
clusters = utils.create_clusters_for_comparing(matrices_propia_10, matrices_SK_10, 10)
utils.plot_SK_Propia_histograms_KMEANS(clusters, 10)

X = nums.datos[:,[i for i in range(nums.datos.shape[1]-1)]]
y = nums.datos[:,-1]
knn=KNeighborsClassifier(5)

#utils.test_knn_SK(pima, wdbc)

kmeans = SKKMeans(10)
a=kmeans.fit(nums.datos)

print(a.labels_)




clusters, centroides = ClusteringKMeans.kMeans(10, nums.datos)
print(centroides)

for a, b in zip(centroides, a.cluster_centers_):
    print(distanciaEuclidea(a[0], b))

"""
clusters = KMeans.kMeans(10, datos1.datos)
confianzas = KMeans.confianzas(clusters, datos1.datos)
media = KMeans.confianzaMedia(confianzas)
for indiceCluster, cluster in clusters.items():
	vp, vn, fp, fn = MC.matrizConfusionCluster(cluster, datos1.datos, confianzas[indiceCluster][1])
	print("Cluster", indiceCluster, "- Clase mayoritaria:", confianzas[indiceCluster][1])
	print("Exactitud:", MC.exactitud(vp, vn, fp, fn))
	print("Precision:", MC.precision(vp, fp))
	print("Sensibilidad:", MC.sensibilidad(vp, fn))
	print("Especificidad:", MC.especificidad(vn, fp), end="\n=================================\n")

#print(datos1.datos[:,1])
#normalizarDatos(datos1.datos, datos1.nominalAtributos)
#print(datos1.datos[:,1])
#datos1 = Datos("ConjuntosDatos/tic-tac-toe.data")
#datos1 = Datos("ConjuntosDatos/german.data")

knn = ClasificadorKNN(25, norm=True)
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


"""