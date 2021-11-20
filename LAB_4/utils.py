from os import X_OK
from Clasificador import ClasificadorNaiveBayes
from ClasificadorRegresionLogistica import ClasificadorRegresionLogistica
from EstrategiaParticionado import ValidacionCruzada, ValidacionSimple
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from ClasificadorKNN import ClasificadorKNN
import matplotlib.pyplot as plt
import MatrizConfusion as mc
import numpy as np


RL_step_1 = 15
RL_step_2 = 100
RL_epoch_END_1 = 100
RL_epoch_END_2 = 100 + RL_step_2

ranges_test = [epoch for epoch in range(10, RL_epoch_END_1, RL_step_1)]
for epoch in range(RL_step_2, RL_epoch_END_2, RL_step_2): ranges_test.append(epoch)

porcentajesTest = [25, 20, 15, 10]
kFoldsTest = [4, 6, 8, 10]
aprendizajes = [0.5, 1.0, 1.5, 2.0]

titulos_histogramas_aprendizaje = ["Cte. Aprendizaje 0.5", "Cte. Aprendizaje 1.0", "Cte. Aprendizaje 1.5", "Cte. Aprendizaje 2.0"]


def RL_test(pima, wdbc):
    errorMedioPimaVC = { epoch: { aprendizaje: [] for aprendizaje in aprendizajes } for epoch in ranges_test }
    errorMedioPimaVS = { epoch: { aprendizaje: [] for aprendizaje in aprendizajes } for epoch in ranges_test }
    errorMedioWDBCVC = { epoch: { aprendizaje: [] for aprendizaje in aprendizajes } for epoch in ranges_test }
    errorMedioWDBCVS = { epoch: { aprendizaje: [] for aprendizaje in aprendizajes } for epoch in ranges_test }

    for epoch in ranges_test:
        print(f"\n*-*-*-**-*-*-**-*-*-**-*-*-**-*-**-*Épocas={epoch}*-*-*-**-*-*-**-*-*-**-*-*-**-*-**-*")
        for aprendizaje in aprendizajes:

            print(f"Test Épocas={epoch} Constante aprendizaje={aprendizaje}\tPima - Error\t\tWDBC - Error")
            
            for porcentaje, kFold in zip(porcentajesTest, kFoldsTest):
                VC = ValidacionCruzada(kFold)
                VS = ValidacionSimple(porcentaje, int(100/porcentaje))

                mediaVC = RL_test_epoch_apren(epoch, aprendizaje, pima, VC)
                mediaVS = RL_test_epoch_apren(epoch, aprendizaje, pima, VS)
                mediaWdbcVC = RL_test_epoch_apren(epoch, aprendizaje, wdbc, VC)
                mediaWdbcVS = RL_test_epoch_apren(epoch, aprendizaje, wdbc, VS)

                errorMedioPimaVC[epoch][aprendizaje].append(mediaVC)
                errorMedioPimaVS[epoch][aprendizaje].append(mediaVS)
                errorMedioWDBCVC[epoch][aprendizaje].append(mediaWdbcVC)
                errorMedioWDBCVS[epoch][aprendizaje].append(mediaWdbcVS)
                
                print(f"Validación Simple {porcentaje}%\t\t\t\t{mediaVS:2f}\t\t{mediaWdbcVS:2f}")
                print(f"Validación Cruzada K-Folds={kFold}\t\t\t{mediaVC:2f}\t\t{mediaWdbcVC:2f}")
            print("================================================================================")
    return errorMedioPimaVC, errorMedioPimaVS, errorMedioWDBCVC, errorMedioWDBCVS


def RL_test_epoch_apren(epoch, apren, dataset, particionado):
    crl = ClasificadorRegresionLogistica(apren, epoch)
    return crl.validacion(particionado, dataset, crl)[0]


def plot_epoch(data, aprendizaje, vs_vc):
    string = "%" if vs_vc else "K-Folds"
    plt.figure(figsize=(20,20))
    test_ranges = porcentajesTest if vs_vc else kFoldsTest
    for i, test_range in enumerate(test_ranges):
        X, Y = ([], [])
        for epoch, values in data.items():
            X.append(epoch)
            Y.append(values[aprendizaje][i])
        plt.subplot(3, 2, i+1)
        plt.plot(X, Y)    
        plt.title(f"Test {test_range}{string}. Regresión logística. Cte. Aprendizaje = {aprendizaje}")
        plt.xlabel("Valor épocas")
        plt.ylabel("Error")


def plot_logistic(data, vs_vc):
    string = "%" if vs_vc else "K-Folds"
    plt.figure(figsize=(20,20))
    test_ranges = porcentajesTest if vs_vc else kFoldsTest
    for i, test_range in enumerate(test_ranges):
        X, Y = ([], [])
        for epoch, values in data.items():
            X.append(epoch)
            Y.append(values[i])
        plt.subplot(3, 2, i+1)
        plt.plot(X, Y)    
        plt.title(f"Test {test_range}{string}. Regresión logística.")
        plt.xlabel("Valor épocas")
        plt.ylabel("Error")


def plot_aprendizaje(data, epoch, vs_vc):
    string = "%" if vs_vc else " K-Folds"
    plt.figure(figsize=(20,20))
    test_ranges = porcentajesTest if vs_vc else kFoldsTest
    for i, test_range in enumerate(test_ranges):
        X, Y = ([i/10 for i in range(5, 25, 5)], 
                [data[epoch][aprendizaje/10][i] for aprendizaje in range(5, 25, 5)])
        plt.subplot(3, 2, i+1)
        plt.plot(X, Y)    
        plt.title(f"Test {test_range}{string}. Regresión logística. Épocas = {epoch}")
        plt.xlabel("Valor aprendizaje")
        plt.ylabel("Error")


def plot_VS_all(datos, aprendizaje, vs_vc, labels):
    string = "%" if vs_vc else "K-Folds"
    plt.figure(figsize=(20,20))
    test_ranges = porcentajesTest if vs_vc else kFoldsTest
    for i, test_range in enumerate(test_ranges):
        plt.subplot(3, 2, i+1)
        for n, data in enumerate(datos):
            X, Y = ([], [])
            for epoch, values in data.items():
                X.append(epoch)
                if "ogistic" in labels[n]:
                    Y.append(values[i])
                else:
                    Y.append(values[aprendizaje][i])
            plt.plot(X, Y, label=labels[n])    

        plt.title(f"Test {test_range}{string}. Regresión logística.")
        plt.xlabel("Valor épocas")
        plt.ylabel("Error")
        plt.legend()


def avg_aprendizaje(data, epoch, aprendizaje):
    return sum(data[epoch][aprendizaje])/len(data[epoch][aprendizaje])


def plot_histograms(datos, epoch):
    plt.figure(figsize=(10,10))
    medias = { i: [] for i in aprendizajes}

    for data in datos:
        for aprendizaje in aprendizajes:
            media = avg_aprendizaje(data, epoch, aprendizaje)
            medias[aprendizaje].append(media)

    for key in medias.keys():
        medias[key] = sum(medias[key])/len(medias[key])
    plt.bar(titulos_histogramas_aprendizaje, medias.values())
    plt.ylabel("Error medio")


def RL_test_SK_logistic(pima, wdbc):
    errorMedioPimaVC = { epoch: [] for epoch in ranges_test }
    errorMedioPimaVS = { epoch: [] for epoch in ranges_test }
    errorMedioWDBCVC = { epoch: [] for epoch in ranges_test }
    errorMedioWDBCVS = { epoch: [] for epoch in ranges_test }

    XPima = pima.datos[:,[i for i in range(pima.datos.shape[1]-1)]]
    XWdbc = wdbc.datos[:,[i for i in range(wdbc.datos.shape[1]-1)]]
    yPima = pima.datos[:,-1]
    yWdbc = wdbc.datos[:,-1]

    for epoch in ranges_test:
        clr = LogisticRegression(max_iter=epoch)
        print(f"\n*-*-*-**-*-*-**-*-*-**-*-*-**-*-**-*Épocas={epoch}*-*-*-**-*-*-**-*-*-**-*-*-**-*-**-*")
        print(f"Test Épocas={epoch}\t\t\t\tPima - Error\t\tWDBC - Error")
            
        for porcentaje, kFold in zip(porcentajesTest, kFoldsTest):
            mediaVC = 1-cross_val_score(clr, XPima, yPima, cv=kFold).mean()
            mediaVS = test_VS(XPima, yPima, int(100/porcentaje), porcentaje/100, clr)
            mediaWdbcVC = 1-cross_val_score(clr, XWdbc, yWdbc, cv=kFold).mean()
            mediaWdbcVS = test_VS(XWdbc, yWdbc, int(100/porcentaje), porcentaje/100, clr)

            errorMedioPimaVC[epoch].append(mediaVC)
            errorMedioPimaVS[epoch].append(mediaVS)
            errorMedioWDBCVC[epoch].append(mediaWdbcVC)
            errorMedioWDBCVS[epoch].append(mediaWdbcVS)
    
            print(f"Validación Simple {porcentaje}%\t\t\t{mediaVS:2f}\t\t{mediaWdbcVS:2f}")
            print(f"Validación Cruzada K-Folds={kFold}\t\t{mediaVC:2f}\t\t{mediaWdbcVC:2f}")
        print("================================================================================")
    return errorMedioPimaVC, errorMedioPimaVS, errorMedioWDBCVC, errorMedioWDBCVS


def RL_test_SK_SGBD(pima, wdbc):
    errorMedioPimaVC = { epoch: { aprendizaje: [] for aprendizaje in aprendizajes } for epoch in ranges_test }
    errorMedioPimaVS = { epoch: { aprendizaje: [] for aprendizaje in aprendizajes } for epoch in ranges_test }
    errorMedioWDBCVC = { epoch: { aprendizaje: [] for aprendizaje in aprendizajes } for epoch in ranges_test }
    errorMedioWDBCVS = { epoch: { aprendizaje: [] for aprendizaje in aprendizajes } for epoch in ranges_test }

    XPima = pima.datos[:,[i for i in range(pima.datos.shape[1]-1)]]
    XWdbc = wdbc.datos[:,[i for i in range(wdbc.datos.shape[1]-1)]]
    yPima = pima.datos[:,-1]
    yWdbc = wdbc.datos[:,-1]

    for epoch in ranges_test:
        print(f"\n*-*-*-**-*-*-**-*-*-**-*-*-**-*-**-*Épocas={epoch}*-*-*-**-*-*-**-*-*-**-*-*-**-*-**-*")
        for aprendizaje in aprendizajes:
            print(f"Test Épocas={epoch} Constante aprendizaje={aprendizaje}\tPima - Error\t\tWDBC - Error")
            
            for porcentaje, kFold in zip(porcentajesTest, kFoldsTest):
                clr = SGDClassifier(max_iter=epoch, learning_rate='constant', eta0=aprendizaje)
                mediaVC = 1-cross_val_score(clr, XPima, yPima, cv=kFold).mean()
                mediaVS = test_VS(XPima, yPima, int(100/porcentaje), porcentaje/100, clr)
                mediaWdbcVC = 1-cross_val_score(clr, XWdbc, yWdbc, cv=kFold).mean()
                mediaWdbcVS = test_VS(XWdbc, yWdbc, int(100/porcentaje), porcentaje/100, clr)

                errorMedioPimaVC[epoch][aprendizaje].append(mediaVC)
                errorMedioPimaVS[epoch][aprendizaje].append(mediaVS)
                errorMedioWDBCVC[epoch][aprendizaje].append(mediaWdbcVC)
                errorMedioWDBCVS[epoch][aprendizaje].append(mediaWdbcVS)
                
                print(f"Validación Simple {porcentaje}%\t\t\t\t{mediaVS:2f}\t\t{mediaWdbcVS:2f}")
                print(f"Validación Cruzada K-Folds={kFold}\t\t\t{mediaVC:2f}\t\t{mediaWdbcVC:2f}")
            print("================================================================================")
    return errorMedioPimaVC, errorMedioPimaVS, errorMedioWDBCVC, errorMedioWDBCVS


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


def espacio_ROC_avg_RL(dataset, times, aprendizaje, epocas, porcentaje):
    tpr_media = []
    fpr_media = []
    for _ in range(times):
        crl = ClasificadorRegresionLogistica(aprendizaje, epocas)
        tpr, fpr = mc.espacioROC(dataset, crl, porcentaje)
        tpr_media.append(tpr)
        fpr_media.append(fpr)
    return sum(tpr_media)/len(tpr_media), sum(fpr_media)/len(fpr_media)


def espacio_ROC_avg_KNN(dataset, times, K, func, porcentaje):
    tpr_media = []
    fpr_media = []
    for _ in range(times):
        cknn = ClasificadorKNN(K, func)
        tpr, fpr = mc.espacioROC(dataset, cknn, porcentaje)
        tpr_media.append(tpr)
        fpr_media.append(fpr)
    return sum(tpr_media)/len(tpr_media), sum(fpr_media)/len(fpr_media)


def espacio_ROC_avg_NB(dataset, times, porcentaje):
    tpr_media = []
    fpr_media = []
    for _ in range(times):
        cnb = ClasificadorNaiveBayes()
        tpr, fpr = mc.espacioROC(dataset, cnb, porcentaje)
        tpr_media.append(tpr)
        fpr_media.append(fpr)
    return sum(tpr_media)/len(tpr_media), sum(fpr_media)/len(fpr_media)


def plot_espacio_ROC(dataset, times, porcentaje, RL_conf, KNN_conf):
    X_RL, Y_RL = espacio_ROC_avg_RL(dataset, times, RL_conf[0], RL_conf[1], porcentaje)
    X_KNN, Y_KNN = espacio_ROC_avg_KNN(dataset, times, KNN_conf[0], KNN_conf[1], porcentaje)
    X_NB, Y_NB = espacio_ROC_avg_NB(dataset, times, porcentaje)
    X = [X_RL, X_KNN, X_NB]
    Y = [Y_RL, Y_KNN, Y_NB]
    labels = ["Regresion logistica", "K-NN", "Naive Bayes"]

    plt.figure(figsize=(10,10))
    _, ax = plt.subplots()
    plt.scatter(X, Y)
    for i, label in enumerate(labels):
        ax,plt.annotate(label, (X[i], Y[i]))
    ax.plot([0, 1], [0, 1], ls="--", c=".1")

    x_ticks = [0, 1]
    y_ticks = [0, 1]
    plt.xticks(x_ticks, x_ticks)
    plt.yticks(y_ticks, y_ticks)
    plt.xlabel("Sensibilidad o TPR")
    plt.xlabel("Especifidad o FPR")
    plt.title(f"Espacio ROC")


def plot_curva_ROC(dataset, aprendizaje, epocas, porcentaje):
    crl = ClasificadorRegresionLogistica(aprendizaje, epocas)

    v = ValidacionSimple(porcentaje, 1)
    v.creaParticiones(dataset.datos)
    crl.entrenamiento(dataset.datos[v.particiones[0].indicesTrain], dataset.nominalAtributos, None)
    tabla = crl.clasificaProbs(dataset.datos[v.particiones[0].indicesTest], dataset.nominalAtributos, None)

    # Eliminamos los que no son ni TP ni FP (no influyen en la gráfica)
    tabla_para_plot = []
    for score, clase, vp, fp in tabla:
        if vp or fp:
            tabla_para_plot.append((score, clase, vp, fp))

    tabla_para_plot = sorted(tabla_para_plot, key=lambda x: x[0], reverse=True)
    X, Y = crea_coordenadas(tabla_para_plot)
    plt.figure(figsize=(20,20))
    plt.plot(X, Y)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"Curva ROC - Regresión logística - Cte. Aprendizaje = {aprendizaje}, épocas = {epocas}")


def crea_coordenadas(tabla):
    X, Y = [], []
    x, y = 0, 0
    for _, _, vp, fp in tabla:
        if vp:
            x += 1
        elif fp:
            y += 1
        X.append(x)
        Y.append(y)
    return normaliza_coordenadas(X, Y)


def normaliza_coordenadas(X, Y):
    nX, nY = [], []
    maxX, maxY = max(X), max(Y)
    for x, y in zip(X, Y):
        nX.append(x/maxX)
        nY.append(y/maxY)
    return nX, nY