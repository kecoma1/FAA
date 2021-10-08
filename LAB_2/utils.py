from Datos import Datos
from Clasificador import ClasificadorNaiveBayes
from EstrategiaParticionado import ValidacionCruzada, ValidacionSimple
import numpy as np

def test(times:int, testPercentage:int, cruzada:bool, data):
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

def testPrint(times:int, testPercentage:int, german, ttt):
	mediaTTTVC, mediaTTTCLPVC, stdTTTVC, stdTTTCLPVC = test(times, testPercentage, True, ttt)
	mediaTTTVS, mediaTTTCLPVS, stdTTTVS, stdTTTCLPVS = test(times, testPercentage, False, ttt)
	mediaGermanVC, mediaGermanCLPVC, stdGermanVC, stdGermanCLPVC = test(times, testPercentage, True, ttt)
	mediaGermanVS, mediaGermanCLPVS, stdGermanVS, stdGermanCLPVS = test(times, testPercentage, False, ttt)
	print(f"Test {testPercentage}% / K = {int(100/testPercentage)}\t\tGerman - Media\tGerman - std\tTic-Tac-Toe - Media\tTic-Tac-Toe - std")
	print(f"Validacion cruzada\t\t{mediaGermanVC:2f}\t{stdGermanVC:2f}\t{mediaTTTVC:2f}\t\t{stdTTTVC:2f}")
	print(f"Validacion cruzada (Laplace)\t{mediaGermanCLPVC:2f}\t{stdGermanCLPVC:2f}\t{mediaTTTCLPVC:2f}\t\t{stdTTTCLPVC:2f}")
	print(f"Validacion simple\t\t{mediaGermanVS:2f}\t{stdGermanVS:2f}\t{mediaTTTVS:2f}\t\t{stdTTTVS:2f}")
	print(f"Validacion simple (Laplace)\t{mediaGermanCLPVS:2f}\t{stdGermanCLPVS:2f}\t{mediaTTTCLPVS:2f}\t\t{stdTTTCLPVS:2f}\n\n")


def testNBPropio():
	german = Datos("ConjuntosDatos/german.data")
	ttt = Datos("ConjuntosDatos/tic-tac-toe.data")
	for i in range(5, 55, 5):
		testPrint(10, i, german, ttt)
