from ClasificadorKNN import ClasificadorKNN
from EstrategiaParticionado import ValidacionCruzada, ValidacionSimple
import numpy as np
import matplotlib.pyplot as plt


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

        print(f"Test K={k} K-NN SIN NORMALIZAR\t\tPima - Error Medio\tPima - std\tWDBC - Error Medio\tWDBC - std")
        
        for porcentaje, kFold in zip(porcentajesTest, kFoldsTest):
            VC = ValidacionCruzada(kFold)
            VS = ValidacionSimple(porcentaje, int(100/porcentaje))

            mediaVC, stdVC = knn_test_k(10, k, False, pima, VC)
            mediaVS, stdVS = knn_test_k(10, k, False, pima, VS)
            mediaWdbcVC, stdWdbcVC = knn_test_k(10, k, False, wdbc, VC)
            mediaWdbcVS, stdWdbcVS = knn_test_k(10, k, False, wdbc, VS)

            errorMedioPimaVC[k].append((mediaVC, stdVC))
            errorMedioPimaVS[k].append((mediaVS, stdVS))
            errorMedioWDBCVC[k].append((mediaWdbcVC, stdWdbcVC))
            errorMedioWDBCVS[k].append((mediaWdbcVS, stdWdbcVS))
            
            print(f"Validación Simple {porcentaje}%\t\t\t{mediaVS:2f}\t\t{stdVS:2f}\t{mediaWdbcVS:2f}\t\t{stdWdbcVS:2f}")
            print(f"Validación Cruzada K-Folds={kFold}\t\t{mediaVC:2f}\t\t{stdVC:2f}\t{mediaWdbcVC:2f}\t\t{stdWdbcVC:2f}")
        print("====================================================================================================================")
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

        print(f"Test K={k} K-NN NORMALIZANDO\tPima - Error Medio\tPima - std\tWDBC - Error Medio\tWDBC - std")
        
        for porcentaje, kFold in zip(porcentajesTest, kFoldsTest):
            VC = ValidacionCruzada(kFold)
            VS = ValidacionSimple(porcentaje, int(100/porcentaje))

            mediaVCNorm, stdVCNorm = knn_test_k(10, k, True, pima, VC)
            mediaVSNorm, stdVSNorm = knn_test_k(10, k, True, pima, VS)
            mediaWdbcVCNorm, stdWdbcVCNorm = knn_test_k(10, k, True, wdbc, VC)
            mediaWdbcVSNorm, stdWdbcVSNorm = knn_test_k(10, k, True, wdbc, VS)

            errorMedioPimaVCNorm[k].append((mediaVCNorm, stdVCNorm))
            errorMedioPimaVSNorm[k].append((mediaVSNorm, stdVSNorm))
            errorMedioWDBCVCNorm[k].append((mediaWdbcVCNorm, stdWdbcVCNorm))
            errorMedioWDBCVSNorm[k].append((mediaWdbcVSNorm, stdWdbcVSNorm))
            
            print(f"Validación Simple {porcentaje}%\t\t\t{mediaVSNorm:2f}\t{stdVSNorm:2f}\t{mediaWdbcVSNorm:2f}\t{stdWdbcVSNorm:2f}")
            print(f"Validación Cruzada K-Folds={kFold}\t{mediaVCNorm:2f}\t{stdVCNorm:2f}\t{mediaWdbcVCNorm:2f}\t{stdWdbcVCNorm:2f}")
        print("====================================================================================================================")
    return errorMedioPimaVCNorm, errorMedioPimaVSNorm, errorMedioWDBCVCNorm, errorMedioWDBCVSNorm


def knn_test_k(times, k, norm, dataset, particionado):
    error = []
    for _ in range(times):
        knn = ClasificadorKNN(k, norm)
        e, _ = knn.validacion(particionado, dataset, knn)
        error.append(e)
    
    error = np.array(error)
    return np.mean(error), np.std(error)


def plot_VS(data, dataNorm):
    plt.figure(figsize=(20,20))
    for i, porcentaje in enumerate(porcentajesTest):
        plt.subplot(3, 2, i+1)
        X = [k for k in range(1, KNN_END_TEST_K, 2)]
        y = [values[i][0] for _, values in data.items()]
        yNorm = [values[i][0] for _, values in dataNorm.items()]
        plt.plot(X, y)
        plt.plot(X, yNorm, label="Datos normalizados")
        plt.title(f"K-NN Validación simple {porcentaje}%")
        plt.xlabel("Valor K")
        plt.ylabel("Error")
        plt.legend()


def plot_VC(data, dataNorm):
    plt.figure(figsize=(20,20))
    for i, kFold in enumerate(kFoldsTest):
        plt.subplot(3, 2, i+1)
        X = [k for k in range(1, KNN_END_TEST_K, 2)]
        y = [values[i][0] for _, values in data.items()]
        yNorm = [values[i][0] for _, values in dataNorm.items()]
        plt.plot(X, y)
        plt.plot(X, yNorm, label="Datos normalizados")
        plt.title(f"K-NN Validación cruzada K={kFold}")
        plt.xlabel("Valor K")
        plt.ylabel("Error")
        plt.legend()


def plot_matriz_confusion(matriz):
    plt.figure(figsize=(20,20))
    plt.bar(titulos_matriz_confusion, matriz)
    plt.ylabel("Porcentaje de la frecuencia")
