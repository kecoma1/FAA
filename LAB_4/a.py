from Clasificador import ClasificadorNaiveBayes
from ClasificadorRegresionLogistica import ClasificadorRegresionLogistica
from EstrategiaParticionado import Particion, ValidacionCruzada, ValidacionSimple
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.model_selection import cross_val_score
from ClasificadorKNN import ClasificadorKNN
from Distancias import distanciaEuclidea
from Datos import Datos
import MatrizConfusion as MC
import utils

wdbc = Datos("ConjuntosDatos/wdbc__a.data")
indians = Datos("ConjuntosDatos/pima-indians-diabetes__a.data")
nums = Datos("ConjuntosDatos/nums.csv")
ttt = Datos("ConjuntosDatos/tic-tac-toe.data")
german = Datos("ConjuntosDatos/german.data")
lent = Datos("ConjuntosDatos/lentillas.data")

errorMedioPimaVC, errorMedioPimaVS, errorMedioWDBCVC, errorMedioWDBCVS = utils.RL_test(indians, wdbc)
utils.plot_epoch(errorMedioPimaVC, 0.5, True)
#crl = ClasificadorRegresionLogistica(1, 10)
#crl.entrenamiento(wdbc.datos, wdbc.nominalAtributos, wdbc.diccionario)
"""
lr = LinearRegression()
X = wdbc.datos[:,[i for i in range(wdbc.datos.shape[1]-1)]]
y = wdbc.datos[:,-1]
print("SKL:", 1-cross_val_score(lr, X, y).mean())

lr = LinearRegression()
X = indians.datos[:,[i for i in range(indians.datos.shape[1]-1)]]
y = indians.datos[:,-1]
print("SKL:", 1-cross_val_score(lr, X, y).mean())

sgdc = SGDClassifier()
X = wdbc.datos[:,[i for i in range(wdbc.datos.shape[1]-1)]]
y = wdbc.datos[:,-1]
print("SKL - SGDClassifier:", 1-cross_val_score(sgdc, X, y).mean())

sgdc = SGDClassifier()
X = indians.datos[:,[i for i in range(indians.datos.shape[1]-1)]]
y = indians.datos[:,-1]
print("SKL - SGDClassifier:", 1-cross_val_score(sgdc, X, y).mean())

crl = ClasificadorRegresionLogistica(1, 1000)
cknn = ClasificadorKNN(35, distanciaEuclidea)
cnb = ClasificadorNaiveBayes()

v = ValidacionSimple(30, 1)


print("RL")
vp, vn, fp, fn = MC.matrizConfusionClasificador(german, crl, 30)
print(MC.TPR(vp, fn))
print(MC.FPR(fp, vn))
print("K-NN")
vp, vn, fp, fn = MC.matrizConfusionClasificador(german, cknn, 30)
print(MC.TPR(vp, fn))
print(MC.FPR(fp, vn))
print("NB")
vp, vn, fp, fn = MC.matrizConfusionClasificador(german, cnb, 30)
print(MC.TPR(vp, fn))
print(MC.FPR(fp, vn))


print(MC.matrizConfusionClasificador(german, crl, 30))
print(MC.matrizConfusionClasificador(german, cknn, 30))
print(MC.matrizConfusionClasificador(german, cnb, 30))

"""

