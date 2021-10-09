from Clasificador import ClasificadorNaiveBayes
from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada, ValidacionSimple
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def test(times: int, testPercentage: int, cruzada: bool, data):
    particionado = ValidacionCruzada(int(100/testPercentage)) if cruzada else ValidacionSimple(testPercentage, int(100/testPercentage))
    cnv1 = ClasificadorNaiveBayes()
    error = []
    errorCLP = []

    for _ in range(times):
        e, eCLP = cnv1.validacion(particionado, data, cnv1)
        error.append(e)
        errorCLP.append(eCLP)

    error = np.array(error)
    errorCLP = np.array(errorCLP)
    return np.mean(error), np.mean(errorCLP), np.std(error), np.std(errorCLP)


def propio_test(german, ttt):
    propioErrorMedioGermanVS = []
    propioErrorMedioGermanVSCLP = []
    propioErrorMedioGermanVC = []
    propioErrorMedioGermanVCCLP = []
    propioErrorMediotttVS = []
    propioErrorMediotttVSCLP = []
    propioErrorMediotttVC = []
    propioErrorMediotttVCCLP = []
    
    for i in range(5, 55, 5):
        mediaTTTVS, mediaTTTCLPVS, stdTTTVS, stdTTTCLPVS = test(10, i, False, ttt)
        mediaGermanVS, mediaGermanCLPVS, stdGermanVS, stdGermanCLPVS = test(10, i, False, german)
        
        propioErrorMedioGermanVS.append(mediaGermanVS)
        propioErrorMedioGermanVSCLP.append(mediaGermanCLPVS)
        propioErrorMediotttVS.append(mediaTTTVS)
        propioErrorMediotttVSCLP.append(mediaTTTCLPVS)
    
        if i < 40:
            mediaTTTVC, mediaTTTCLPVC, stdTTTVC, stdTTTCLPVC = test(10, int(100/i), True, ttt)
            mediaGermanVC, mediaGermanCLPVC, stdGermanVC, stdGermanCLPVC = test(10, int(100/i), True, german)

            propioErrorMedioGermanVC.append(mediaGermanVC)
            propioErrorMedioGermanVCCLP.append(mediaGermanCLPVC)
            propioErrorMediotttVC.append(mediaTTTVC)
            propioErrorMediotttVCCLP.append(mediaTTTCLPVC)
            
            print(f"Test {i}% / K = {int(100/i)}\t\tGerman - Media\tGerman - std\tTic-Tac-Toe - Media\tTic-Tac-Toe - std")
            print(f"Validacion cruzada\t\t{mediaGermanVC:2f}\t{stdGermanVC:2f}\t{mediaTTTVC:2f}\t\t{stdTTTVC:2f}")
            print(f"Validacion cruzada (Laplace)\t{mediaGermanCLPVC:2f}\t{stdGermanCLPVC:2f}\t{mediaTTTCLPVC:2f}\t\t{stdTTTCLPVC:2f}")
        else:
            print(f"Test {i}%\t\t\tGerman - Media\tGerman - std\tTic-Tac-Toe - Media\tTic-Tac-Toe - std")
        print(f"Validacion simple\t\t{mediaGermanVS:2f}\t{stdGermanVS:2f}\t{mediaTTTVS:2f}\t\t{stdTTTVS:2f}")
        print(f"Validacion simple (Laplace)\t{mediaGermanCLPVS:2f}\t{stdGermanCLPVS:2f}\t{mediaTTTCLPVS:2f}\t\t{stdTTTCLPVS:2f}\n\n")

    return propioErrorMedioGermanVS, propioErrorMedioGermanVSCLP, propioErrorMedioGermanVC, propioErrorMedioGermanVCCLP, propioErrorMediotttVS, propioErrorMediotttVSCLP, propioErrorMediotttVC, propioErrorMediotttVCCLP


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


def test_VS2(X, y, times, testSize, model):
    errores = np.zeros(times)
    for i in range(times):
        _, XTest, _, yTest = train_test_split(X, y, test_size=testSize)
        model.fit(X, y)
        yPred = model.predict(XTest)
        errores[i] = error(yTest, yPred)
    return 1-np.mean(errores), np.std(errores)


def multinomial_test(german, ttt):
	tttX = ttt.datos[:,[i for i in range(ttt.datos.shape[1]-1)]]
	ttty = ttt.datos[:,-1]
	germanX = german.datos[:,[i for i in range(german.datos.shape[1]-1)]]
	germany = german.datos[:,-1]

	MNBerrorMedioGermanVS = []
	MNBerrorMedioGermanVSCLP = []
	MNBerrorMedioGermanVC = []
	MNBerrorMedioGermanVCCLP = []
	MNBerrorMediotttVS = []
	MNBerrorMediotttVSCLP = []
	MNBerrorMediotttVC = []
	MNBerrorMediotttVCCLP = []

	for i in range(5, 55, 5):
		mnb = MultinomialNB(fit_prior=True)
		mnbCLP = MultinomialNB(fit_prior=True, alpha=1)
		
		tttScoreVS = test_VS(tttX, ttty, 10, i/100, mnb)
		tttScoreVSCLP = test_VS(tttX, ttty, 10, i/100, mnbCLP)
		germanScoreVS = test_VS(germanX, germany, 10, i/100, mnb)
		germanScoreVSCLP = test_VS(germanX, germany, 10, i/100, mnbCLP)

		MNBerrorMedioGermanVS.append(germanScoreVS[0])
		MNBerrorMedioGermanVSCLP.append(germanScoreVSCLP[0])
		MNBerrorMediotttVS.append(tttScoreVS[0])
		MNBerrorMediotttVSCLP.append(tttScoreVSCLP[0])
		
		if i < 40:
			tttScore = cross_val_score(mnb, tttX, ttty, cv=int(100/i))
			tttScoreCLP = cross_val_score(mnbCLP, tttX, ttty, cv=int(100/i))
			germanScore = cross_val_score(mnb, germanX, germany, cv=int(100/i))
			germanScoreCLP = cross_val_score(mnbCLP, germanX, germany, cv=int(100/i))
			
			MNBerrorMedioGermanVC.append(1-germanScore.mean())
			MNBerrorMedioGermanVCCLP.append(1-germanScoreCLP.mean())
			MNBerrorMediotttVC.append(1-tttScore.mean())
			MNBerrorMediotttVCCLP.append(1-tttScoreCLP.mean())
			
			print(f"Test {i}% / K = {int(100/i)}\t\tGerman - Media\tGerman - std\tTic-Tac-Toe - Media\tTic-Tac-Toe - std")
			print(f"Validacion cruzada\t\t{1-germanScore.mean():2f}\t{germanScore.std():2f}\t{1-tttScore.mean():2f}\t\t{tttScore.std():2f}")
			print(f"Validacion cruzada (Laplace)\t{1-germanScoreCLP.mean():2f}\t{germanScoreCLP.std():2f}\t{1-tttScoreCLP.mean():2f}\t\t{tttScoreCLP.std():2f}")
		else:
			print(f"Test {i}%\t\t\tGerman - Media\tGerman - std\tTic-Tac-Toe - Media\tTic-Tac-Toe - std")
		print(f"Validacion simple\t\t{germanScoreVS[0]:2f}\t{germanScoreVS[1]:2f}\t{tttScoreVS[0]:2f}\t\t{tttScoreVS[1]:2f}")
		print(f"Validacion simple (Laplace)\t{germanScoreVSCLP[0]:2f}\t{germanScoreVSCLP[1]:2f}\t{tttScoreVSCLP[0]:2f}\t\t{tttScoreVSCLP[1]:2f}\n\n")
	return MNBerrorMedioGermanVS, MNBerrorMedioGermanVSCLP, MNBerrorMedioGermanVC, MNBerrorMedioGermanVCCLP, MNBerrorMediotttVS, MNBerrorMediotttVSCLP, MNBerrorMediotttVC, MNBerrorMediotttVCCLP


def gaussian_test(german, ttt):
    tttX = ttt.datos[:,[i for i in range(ttt.datos.shape[1]-1)]]
    ttty = ttt.datos[:,-1]
    germanX = german.datos[:,[i for i in range(german.datos.shape[1]-1)]]
    germany = german.datos[:,-1]
    
    GNBerrorMedioGermanVS = []
    GNBerrorMedioGermanVC = []
    GNBerrorMediotttVS = []
    GNBerrorMediotttVC = []

    for i in range(5, 55, 5):
        gnb = GaussianNB()
        
        tttScoreVS = test_VS(tttX, ttty, 10, i/100, gnb)
        germanScoreVS = test_VS(germanX, germany, 10, i/100, gnb)
        
        GNBerrorMedioGermanVS.append(germanScoreVS[0])
        GNBerrorMediotttVS.append(tttScoreVS[0])
        
        if i < 40:
            tttScore = cross_val_score(gnb, tttX, ttty, cv=int(100/i))
            germanScore = cross_val_score(gnb, germanX, germany, cv=int(100/i))
            
            GNBerrorMedioGermanVC.append(1-germanScore.mean())
            GNBerrorMediotttVC.append(1-tttScore.mean())
            
            print(f"Test {i}% / K = {int(100/i)}\t\tGerman - Media\tGerman - std\tTic-Tac-Toe - Media\tTic-Tac-Toe - std")
            print(f"Validacion cruzada\t\t{1-germanScore.mean():2f}\t{germanScore.std():2f}\t{1-tttScore.mean():2f}\t\t{tttScore.std():2f}")
        else:
            print(f"Test {i}%\t\t\tGerman - Media\tGerman - std\tTic-Tac-Toe - Media\tTic-Tac-Toe - std")
        print(f"Validacion simple\t\t{1-germanScoreVS[0]:2f}\t{germanScoreVS[1]:2f}\t{tttScoreVS[0]:2f}\t\t{tttScoreVS[1]:2f}\n")
    return GNBerrorMedioGermanVS, GNBerrorMedioGermanVC, GNBerrorMediotttVS, GNBerrorMediotttVC


def categorical_test(german, ttt):
    tttX = ttt.datos[:,[i for i in range(ttt.datos.shape[1]-1)]]
    ttty = ttt.datos[:,-1]
    germanX = german.datos[:,[i for i in range(german.datos.shape[1]-1)]]
    germany = german.datos[:,-1]

    CNBerrorMedioGermanVS = []
    CNBerrorMedioGermanVSCLP = []
    CNBerrorMediotttVS = []
    CNBerrorMediotttVSCLP = []

    for i in range(5, 55, 5):
        cnb = CategoricalNB(fit_prior=True)
        cnbCLP = CategoricalNB(fit_prior=True, alpha=1)
        
        tttScoreVS = test_VS2(tttX, ttty, 10, i/100, cnb)
        tttScoreVSCLP = test_VS2(tttX, ttty, 10, i/100, cnbCLP)
        germanScoreVS = test_VS2(germanX, germany, 10, i/100, cnb)
        germanScoreVSCLP = test_VS2(germanX, germany, 10, i/100, cnbCLP)
        #germanScore = cross_val_score(cnb, germanX, germany, cv=int(100/i))
        #germanScoreCLP = cross_val_score(cnbCLP, germanX, germany, cv=int(100/i))
        
        CNBerrorMedioGermanVS.append(germanScoreVS[0])
        CNBerrorMedioGermanVSCLP.append(germanScoreVSCLP[0])
        CNBerrorMediotttVS.append(tttScoreVS[0])
        CNBerrorMediotttVSCLP.append(tttScoreVSCLP[0])

        print(f"Test {i}%\t\t\tGerman - Media\tGerman - std\tTic-Tac-Toe - Media\tTic-Tac-Toe - std")
        
        tttScore = cross_val_score(cnb, tttX, ttty, cv=int(100/i))
        tttScoreCLP = cross_val_score(cnbCLP, tttX, ttty, cv=int(100/i))
        print(f"Validacion simple\t\t{1-germanScoreVS[0]:2f}\t{germanScoreVS[1]:2f}\t{tttScoreVS[0]:2f}\t\t{tttScoreVS[1]:2f}")
        print(f"Validacion simple (Laplace)\t{1-germanScoreVSCLP[0]:2f}\t{germanScoreVSCLP[1]:2f}\t{tttScoreVSCLP[0]:2f}\t\t{tttScoreVSCLP[1]:2f}\n\n")

    return CNBerrorMedioGermanVS, CNBerrorMedioGermanVSCLP, CNBerrorMediotttVS, CNBerrorMediotttVSCLP

def plot_CLP_MNB(data, titulos):
    VSRange = [i for i in range(5,55,5)]
    VCRange = [20, 10, 6, 5, 4, 3, 2]

    plt.figure(figsize=(15,15))
    for i, tupla in enumerate(data):
        plt.subplot(3, 2, i+1)
        X = VSRange if i % 2 == 0 else VCRange
        plt.plot(X, tupla[0], label='Sin correccion de laplace')
        plt.plot(X, tupla[1], label='Correción de laplace')
        plt.xlabel("% para el test")
        plt.ylabel("Error")
        plt.title(titulos[i])
        plt.legend()