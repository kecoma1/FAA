from ClasificadorRegresionLogistica import ClasificadorRegresionLogistica
from EstrategiaParticionado import ValidacionCruzada, ValidacionSimple
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt


RL_step_1 = 15
RL_step_2 = 100
RL_step_aprendizaje = 5 # LUEGO SE DIVIDE ENTRE 10
RL_epoch_END_1 = 100
RL_epoch_END_2 = 100 + RL_step_2
RL_aprendizaje_END = 25 # LUEGO SE DIVIDE ENTRE 10

ranges_test = [epoch for epoch in range(10, RL_epoch_END_1, RL_step_1)]
for epoch in range(RL_step_2, RL_epoch_END_2, RL_step_2): ranges_test.append(epoch)

porcentajesTest = [25, 20, 15, 10]
kFoldsTest = [4, 6, 8, 10]
aprendizajes = [0.5, 1.0, 1.5, 2.0]

titulos_histogramas_aprendizaje = ["Cte. Aprendizaje 0.5", "Cte. Aprendizaje 1.0", "Cte. Aprendizaje 1.5", "Cte. Aprendizaje 2.0"]


def RL_test(pima, wdbc):
    errorMedioPimaVC = { epoch: { (aprendizaje/10): [] for aprendizaje in range(RL_step_aprendizaje, RL_aprendizaje_END, RL_step_aprendizaje) } for epoch in ranges_test }
    errorMedioPimaVS = { epoch: { (aprendizaje/10): [] for aprendizaje in range(RL_step_aprendizaje, RL_aprendizaje_END, RL_step_aprendizaje) } for epoch in ranges_test }
    errorMedioWDBCVC = { epoch: { (aprendizaje/10): [] for aprendizaje in range(RL_step_aprendizaje, RL_aprendizaje_END, RL_step_aprendizaje) } for epoch in ranges_test }
    errorMedioWDBCVS = { epoch: { (aprendizaje/10): [] for aprendizaje in range(RL_step_aprendizaje, RL_aprendizaje_END, RL_step_aprendizaje) } for epoch in ranges_test }

    for epoch in ranges_test:
        print(f"\n*-*-*-**-*-*-**-*-*-**-*-*-**-*-**-*Épocas={epoch}*-*-*-**-*-*-**-*-*-**-*-*-**-*-**-*")
        for aprendizaje in range(RL_step_aprendizaje, RL_aprendizaje_END, RL_step_aprendizaje):
            aprendizaje /= 10

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


def plot_aprendizaje(data, epoch, vs_vc):
    string = "%" if vs_vc else " K-Folds"
    plt.figure(figsize=(20,20))
    test_ranges = porcentajesTest if vs_vc else kFoldsTest
    for i, test_range in enumerate(test_ranges):
        X, Y = ([i/10 for i in range(5, 25, 5)], 
                [data[epoch][aprendizaje/10] for aprendizaje in range(5, 25, 5)])
        plt.subplot(3, 2, i+1)
        plt.plot(X, Y)    
        plt.title(f"Test {test_range}{string}. Regresión logística. Épocas = {epoch}")
        plt.xlabel("Valor aprendizaje")
        plt.ylabel("Error")


def avg_aprendizaje(data, epoch, aprendizaje):
    return sum(data[epoch][aprendizaje])/len(data[epoch][aprendizaje])


def plot_histograms(datos, epoch):
    plt.figure(figsize=(20,20))
    medias = { i/10: [] for i in range(5, 25, 5)}
    aprendizajes

    for data in datos:
        for i, aprendizaje in enumerate(aprendizajes):
            media = avg_aprendizaje(data, epoch, aprendizaje)
            medias[i] += media
        for i in range(4):
            medias[i] /= 4
    plt.bar(titulos_histogramas_aprendizaje, medias)
    plt.ylabel("Error")