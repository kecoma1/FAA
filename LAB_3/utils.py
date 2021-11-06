from sklearn.model_selection import cross_val_score, train_test_split
from EstrategiaParticionado import ValidacionCruzada, ValidacionSimple
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans as SKKMeans
from ClasificadorKNN import ClasificadorKNN
from Distancias import distanciaEuclidea
import matplotlib.pyplot as plt
import MatrizConfusion as MC
import numpy as np
import Datos
import KMeans


porcentajesTest = [25, 20, 15, 10]
kFoldsTest = [4, 6, 8, 10]
KNN_END_TEST_K = 37
titulos_matriz_confusion = ["Verdaderos Positivos", "Verdaderos Negativos", "Falsos Positivos", "Falsos Negativos"]


def knn_test(pima, wdbc):
    errorMedioPimaVC = {}
    errorMedioPimaVS = {}
    errorMedioWDBCVC = {}
    errorMedioWDBCVS = {}

    for k in range(1, KNN_END_TEST_K, 2):

        errorMedioPimaVC[k] = []
        errorMedioPimaVS[k] = []
        errorMedioWDBCVC[k] = []
        errorMedioWDBCVS[k] = []

        print(f"Test K={k} K-NN SIN NORMALIZAR\t\tPima - Error\t\tWDBC - Error")
        
        for porcentaje, kFold in zip(porcentajesTest, kFoldsTest):
            VC = ValidacionCruzada(kFold)
            VS = ValidacionSimple(porcentaje, int(100/porcentaje))

            mediaVC = knn_test_k(k, False, pima, VC)
            mediaVS = knn_test_k(k, False, pima, VS)
            mediaWdbcVC = knn_test_k(k, False, wdbc, VC)
            mediaWdbcVS = knn_test_k(k, False, wdbc, VS)

            errorMedioPimaVC[k].append(mediaVC)
            errorMedioPimaVS[k].append(mediaVS)
            errorMedioWDBCVC[k].append(mediaWdbcVC)
            errorMedioWDBCVS[k].append(mediaWdbcVS)
            
            print(f"Validación Simple {porcentaje}%\t\t\t{mediaVS:2f}\t\t{mediaWdbcVS:2f}")
            print(f"Validación Cruzada K-Folds={kFold}\t\t{mediaVC:2f}\t\t{mediaWdbcVC:2f}")
        print("================================================================================")
    return errorMedioPimaVC, errorMedioPimaVS, errorMedioWDBCVC, errorMedioWDBCVS


def knn_test_norm(pima, wdbc):
    errorMedioPimaVCNorm = {}
    errorMedioPimaVSNorm = {}
    errorMedioWDBCVCNorm = {}
    errorMedioWDBCVSNorm = {}

    for k in range(1, KNN_END_TEST_K, 2):
        errorMedioPimaVCNorm[k] = []
        errorMedioPimaVSNorm[k] = []
        errorMedioWDBCVCNorm[k] = []
        errorMedioWDBCVSNorm[k] = []

        print(f"Test K={k} K-NN NORMALIZANDO\t\tPima - Error\t\tWDBC - Error")
        
        for porcentaje, kFold in zip(porcentajesTest, kFoldsTest):
            VC = ValidacionCruzada(kFold)
            VS = ValidacionSimple(porcentaje, int(100/porcentaje))

            mediaVCNorm = knn_test_k(k, True, pima, VC)
            mediaVSNorm = knn_test_k(k, True, pima, VS)
            mediaWdbcVCNorm = knn_test_k(k, True, wdbc, VC)
            mediaWdbcVSNorm = knn_test_k(k, True, wdbc, VS)

            errorMedioPimaVCNorm[k].append(mediaVCNorm)
            errorMedioPimaVSNorm[k].append(mediaVSNorm)
            errorMedioWDBCVCNorm[k].append(mediaWdbcVCNorm)
            errorMedioWDBCVSNorm[k].append(mediaWdbcVSNorm)
            
            print(f"Validación Simple {porcentaje}%\t\t\t{mediaVSNorm:2f}\t\t{mediaWdbcVSNorm:2f}")
            print(f"Validación Cruzada K-Folds={kFold}\t\t{mediaVCNorm:2f}\t\t{mediaWdbcVCNorm:2f}")
        print("================================================================================")
    return errorMedioPimaVCNorm, errorMedioPimaVSNorm, errorMedioWDBCVCNorm, errorMedioWDBCVSNorm


def knn_test_k(k, norm, dataset, particionado):
    knn = ClasificadorKNN(k, distanciaEuclidea, norm)
    e, _ = knn.validacion(particionado, dataset, knn)
    error = e
    return error


def plot_VS(data, dataNorm):
    plt.figure(figsize=(20,20))
    for i, porcentaje in enumerate(porcentajesTest):
        plt.subplot(3, 2, i+1)
        X = [k for k in range(len(data))]
        y = [values[i] for _, values in data.items()]
        yNorm = [values[i] for _, values in dataNorm.items()]
        plt.plot(X, y)
        plt.plot(X, yNorm, label="Datos normalizados")
        plt.title(f"K-NN Validación simple {porcentaje}%")
        plt.xlabel("Valor K")
        plt.ylabel("Error")
        plt.legend()


def plot_VS_SK_Propio(dataP, dataNormP, dataSK, dataNormSK):
    plt.figure(figsize=(20,20))
    for i, porcentaje in enumerate(porcentajesTest):
        plt.subplot(3, 2, i+1)
        X = [k for k in range(len(dataP))]
        yP = [values[i] for _, values in dataP.items()]
        ySK = [values[i] for _, values in dataSK.items()]
        yNormP = [values[i] for _, values in dataNormP.items()]
        yNormSK = [values[i] for _, values in dataNormSK.items()]
        plt.plot(X, yP, label="KNN Propio")
        plt.plot(X, yNormP, label="Norm. KNN Propio")
        plt.plot(X, ySK, label="KNN SK")
        plt.plot(X, yNormSK, label="Norm. KNN SK")
        plt.title(f"K-NN Validación simple {porcentaje}%")
        plt.xlabel("Valor K")
        plt.ylabel("Error")
        plt.legend()


def plot_VC(data, dataNorm):
    plt.figure(figsize=(20,20))
    for i, kFold in enumerate(kFoldsTest):
        plt.subplot(3, 2, i+1)
        X = [k for k in range(len(data))]
        y = [values[i] for _, values in data.items()]
        yNorm = [values[i] for _, values in dataNorm.items()]
        plt.plot(X, y)
        plt.plot(X, yNorm, label="Datos normalizados")
        plt.title(f"K-NN Validación cruzada K={kFold}")
        plt.xlabel("Valor K")
        plt.ylabel("Error")
        plt.legend()


def plot_VC_SK_Propio(dataP, dataNormP, dataSK, dataNormSK):
    plt.figure(figsize=(20,20))
    for i, kFold in enumerate(kFoldsTest):
        plt.subplot(3, 2, i+1)
        X = [k for k in range(len(dataP))]
        yP = [values[i] for _, values in dataP.items()]
        ySK = [values[i] for _, values in dataSK.items()]
        yNormP = [values[i] for _, values in dataNormP.items()]
        yNormSK = [values[i] for _, values in dataNormSK.items()]
        plt.plot(X, yP, label="KNN Propio")
        plt.plot(X, yNormP, label="Norm. KNN Propio")
        plt.plot(X, ySK, label="KNN SK")
        plt.plot(X, yNormSK, label="Norm. KNN SK")
        plt.title(f"K-NN Validación cruzada K={kFold}")
        plt.xlabel("Valor K")
        plt.ylabel("Error")
        plt.legend()


def test_KMeans(k, data):
    X = data.datos[:,[i for i in range(data.datos.shape[1]-1)]]
    clusters, centroides = KMeans.kMeans(k, X)
    confianzas = KMeans.confianzas(clusters, data.datos)
    plt.figure(figsize=(25, 35))
    matrices = []
    for i, (indiceCluster, cluster) in enumerate(clusters.items()):
        vp, vn, fp, fn = MC.matrizConfusionCluster(cluster, data.datos, confianzas[indiceCluster][1])
        plt.subplot(int(k/2)+1, 2, i+1)
        plt.bar(titulos_matriz_confusion, [vp, vn, fp, fn])    
        plt.title(f"Cluster {indiceCluster+1} - Clase mayoritaria '{confianzas[indiceCluster][1]}'")
        plt.ylabel("Porcentaje de la frecuencia")
        matrices.append([(vp, vn, fp, fn), confianzas[indiceCluster][1]])
    return matrices, centroides


def test_KMeans_SK(k, data):
    X = data.datos[:,[i for i in range(data.datos.shape[1]-1)]]
    kmeans = SKKMeans(k)
    res = kmeans.fit(X)
    clusters = KMeans.get_SK_clusters(res.labels_, k)
    confianzas = KMeans.confianzas(clusters, data.datos)
    plt.figure(figsize=(25, 35))
    matrices = []
    for i, (indiceCluster, cluster) in enumerate(clusters.items()):
        vp, vn, fp, fn = MC.matrizConfusionCluster(cluster, data.datos, confianzas[indiceCluster][1])
        plt.subplot(int(k/2)+1, 2, i+1)
        plt.bar(titulos_matriz_confusion, [vp, vn, fp, fn])    
        plt.title(f"SKLearn - Cluster {indiceCluster+1} - Clase mayoritaria '{confianzas[indiceCluster][1]}'")
        plt.ylabel("Porcentaje de la frecuencia")
        matrices.append([(vp, vn, fp, fn), confianzas[indiceCluster][1]])
    return matrices, res.cluster_centers_


def create_clusters_for_comparing(propia, SK, k):
    clusters = {i : [] for i in range(k)}
    for matriz, clase in propia:
            clusters[clase].append((matriz, "propia"))

    for matriz, clase in SK:
        clusters[clase].append((matriz, "SK"))
    
    return clusters  


def plot_SK_Propia_histograms_KMEANS(clusters, k):
    plt.figure(figsize=(25, 35))
    for i, (clase, matrices) in enumerate(clusters.items()):
        plt.subplot(int(k/2)+1, 2, i+1)
        vps = []
        vns = []
        fps = []
        fns = [] 

        for matriz, tipo in matrices:
            vps.append((matriz[0], tipo))
            vns.append((matriz[1], tipo))
            fps.append((matriz[2], tipo))
            fns.append((matriz[3], tipo))

        for vp, tipo in vps:
            titulo = "VP SKLearn" if tipo == "SK" else "VP Propia"
            plt.bar(titulo, vp, alpha=0.5) 

        for vn, tipo in vns:
            titulo = "VN SKLearn" if tipo == "SK" else "VN Propia"
            plt.bar(titulo, vn) 

        for fp, tipo in fps:
            titulo = "FP SKLearn" if tipo == "SK" else "FP Propia"
            plt.bar(titulo, fp) 

        for fn, tipo in fns:
            titulo = "FN SKLearn" if tipo == "SK" else "FN Propia"
            plt.bar(titulo, fn)    

            plt.title(f"SKLearn - Cluster {i+1} - Clase mayoritaria '{clase}'")
        plt.ylabel("Porcentaje de la frecuencia")    


def print_matriz_confusion(matriz, titulo):
    print("\n"+titulo)
    print("\t\tPositivos-Negativos")
    print("\t\t+-------+-------+")
    print(f"Verdaderos\t+ {matriz[0]}\t+ {matriz[1]}\t+")
    print("\t\t+-------+-------+")
    print(f"Falsos\t\t+ {matriz[2]}\t+ {matriz[3]}\t+")
    print("\t\t+-------+-------+")


def tablas_matriz_confusion(matrices):
    for i, matriz in enumerate(matrices):
        print_matriz_confusion(matriz[0], f"Cluster {i+1} - Clase mayoritaria '{matriz[1]}'")


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
    return np.mean(errores)


def knn_test_SK(pima, wdbc):
    XPima = pima.datos[:,[i for i in range(pima.datos.shape[1]-1)]]
    XWdbc = wdbc.datos[:,[i for i in range(wdbc.datos.shape[1]-1)]]
    yPima = pima.datos[:,-1]
    yWdbc = wdbc.datos[:,-1]

    errorMedioPimaVC_SK = {}
    errorMedioPimaVS_SK = {}
    errorMedioWDBCVC_SK = {}
    errorMedioWDBCVS_SK = {}

    for k in range(1, KNN_END_TEST_K, 2):

        knn = KNeighborsClassifier(k)

        errorMedioPimaVC_SK[k] = []
        errorMedioPimaVS_SK[k] = []
        errorMedioWDBCVC_SK[k] = []
        errorMedioWDBCVS_SK[k] = []

        print(f"Test K={k} K-NN SKLearn SIN NORMALIZAR\tPima - Error\t\tWDBC - Error")
        
        for porcentaje, kFold in zip(porcentajesTest, kFoldsTest):

            mediaVC = 1-cross_val_score(knn, XPima, yPima, cv=kFold).mean()
            mediaVS = test_VS(XPima, yPima, int(100/porcentaje), porcentaje/100, knn)
            mediaWdbcVC = 1-cross_val_score(knn, XWdbc, yWdbc, cv=kFold).mean()
            mediaWdbcVS = test_VS(XWdbc, yWdbc, int(100/porcentaje), porcentaje/100, knn)

            errorMedioPimaVC_SK[k].append(mediaVC)
            errorMedioPimaVS_SK[k].append(mediaVS)
            errorMedioWDBCVC_SK[k].append(mediaWdbcVC)
            errorMedioWDBCVS_SK[k].append(mediaWdbcVS)
            
            print(f"Validación Simple {porcentaje}%\t\t\t{mediaVS:2f}\t\t{mediaWdbcVS:2f}")
            print(f"Validación Cruzada K-Folds={kFold}\t\t{mediaVC:2f}\t\t{mediaWdbcVC:2f}")
        print("================================================================================")
    return errorMedioPimaVC_SK, errorMedioPimaVS_SK, errorMedioWDBCVC_SK, errorMedioWDBCVS_SK


def knn_test_norm_SK(pima, wdbc):
    pimaNorm = Datos.normalizarDatos(pima.datos, pima.nominalAtributos)
    wdbcNorm = Datos.normalizarDatos(wdbc.datos, wdbc.nominalAtributos)

    XPima = pimaNorm[:,[i for i in range(pimaNorm.shape[1]-1)]]
    XWdbc = wdbcNorm[:,[i for i in range(wdbcNorm.shape[1]-1)]]
    yPima = pimaNorm[:,-1]
    yWdbc = wdbcNorm[:,-1]

    errorMedioPimaVCNorm_SK = {}
    errorMedioPimaVSNorm_SK = {}
    errorMedioWDBCVCNorm_SK = {}
    errorMedioWDBCVSNorm_SK = {}

    for k in range(1, KNN_END_TEST_K, 2):

        knn = KNeighborsClassifier(k)

        errorMedioPimaVCNorm_SK[k] = []
        errorMedioPimaVSNorm_SK[k] = []
        errorMedioWDBCVCNorm_SK[k] = []
        errorMedioWDBCVSNorm_SK[k] = []

        print(f"Test K={k} K-NN SKLearn NORMALIZANDO\tPima - Error\t\tWDBC - Error")
        
        for porcentaje, kFold in zip(porcentajesTest, kFoldsTest):

            mediaVCNorm = 1-cross_val_score(knn, XPima, yPima, cv=kFold).mean()
            mediaVSNorm = test_VS(XPima, yPima, int(100/porcentaje), porcentaje/100, knn)
            mediaWdbcVCNorm = 1-cross_val_score(knn, XWdbc, yWdbc, cv=kFold).mean()
            mediaWdbcVSNorm = test_VS(XWdbc, yWdbc, int(100/porcentaje), porcentaje/100, knn)

            errorMedioPimaVCNorm_SK[k].append(mediaVCNorm)
            errorMedioPimaVSNorm_SK[k].append(mediaVSNorm)
            errorMedioWDBCVCNorm_SK[k].append(mediaWdbcVCNorm)
            errorMedioWDBCVSNorm_SK[k].append(mediaWdbcVSNorm)
            
            print(f"Validación Simple {porcentaje}%\t\t\t{mediaVSNorm:2f}\t\t{mediaWdbcVSNorm:2f}")
            print(f"Validación Cruzada K-Folds={kFold}\t\t{mediaVCNorm:2f}\t\t{mediaWdbcVCNorm:2f}")
        print("================================================================================")
    return errorMedioPimaVCNorm_SK, errorMedioPimaVSNorm_SK, errorMedioWDBCVCNorm_SK, errorMedioWDBCVSNorm_SK


