from numpy.core.fromnumeric import shape
from Datos import Datos
from scipy.stats import norm
import numpy as np
from ClasificadorKNN import ClasificadorKNN
from EstrategiaParticionado import ValidacionCruzada, ValidacionSimple
import KMeans
import MatrizConfusion as MC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import utils

nums = Datos("ConjuntosDatosP2KMeans/nums.csv")
wdbc = Datos("ConjuntosDatosP2Knn/wdbc.data")
pima = Datos("ConjuntosDatosP2Knn/pima-indians-diabetes.data")

def error(datos, pred):
    errores = 0
    for i in range(datos.shape[0]):
        if datos[i] != pred[i]:
            errores += 1
    return (errores/datos.shape[0])

def test_VS(X, y, times, testSize, model):
    errores = np.zeros(times)
    for i in range(times):
        XTrain, XTest, yTrain, yTest = train_test_split(
            X, y, test_size=testSize)
        model.fit(XTrain, yTrain)
        yPred = model.predict(XTest)
        errores[i] = error(yTest, yPred)
    return np.mean(errores), np.std(errores)

X = pima.datos[:,[i for i in range(pima.datos.shape[1]-1)]]
y = pima.datos[:,-1]
knn=KNeighborsClassifier(5)

#utils.test_knn_SK(pima, wdbc)
utils.test_knn_norm_SK(pima, wdbc)



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