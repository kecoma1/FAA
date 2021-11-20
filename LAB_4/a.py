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

# Para esconder los warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

wdbc = Datos("ConjuntosDatos/wdbc__a.data")
indians = Datos("ConjuntosDatos/pima-indians-diabetes__a.data")
nums = Datos("ConjuntosDatos/nums.csv")
ttt = Datos("ConjuntosDatos/tic-tac-toe.data")
german = Datos("ConjuntosDatos/german.data")
lent = Datos("ConjuntosDatos/lentillas.data")

RL_conf = (1, 1000)
KNN_conf = (11, distanciaEuclidea)
utils.plot_espacio_ROC(wdbc, 2, 20, RL_conf, KNN_conf)
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
vp_crl, vn_crl, fp_crl, fn_crl = MC.matrizConfusionClasificador(german, crl, 30)
print(MC.TPR(vp, fn))
print(MC.FPR(fp, vn))
print("K-NN")
vp_knn, vn_knn, fp_knn, fn_knn = MC.matrizConfusionClasificador(german, cknn, 30)
print(MC.TPR(vp, fn))
print(MC.FPR(fp, vn))
print("NB")
vp_nb, vn_nb, fp_nb, fn_nb = MC.matrizConfusionClasificador(german, cnb, 30)
print(MC.TPR(vp, fn))
print(MC.FPR(fp, vn))


print(MC.matrizConfusionClasificador(german, crl, 30))
print(MC.matrizConfusionClasificador(german, cknn, 30))
print(MC.matrizConfusionClasificador(german, cnb, 30))

"""

