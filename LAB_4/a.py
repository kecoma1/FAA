from ClasificadorRegresionLogistica import ClasificadorRegresionLogistica
from EstrategiaParticionado import ValidacionCruzada, ValidacionSimple
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.model_selection import cross_val_score
from Datos import Datos

wdbc = Datos("ConjuntosDatos/wdbc.data")
indians = Datos("ConjuntosDatos/pima-indians-diabetes.data")

#crl = ClasificadorRegresionLogistica(1, 10)
#crl.entrenamiento(wdbc.datos, wdbc.nominalAtributos, wdbc.diccionario)

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


vc = ValidacionCruzada(10)
vs = ValidacionSimple(10, 10)



crl = ClasificadorRegresionLogistica(1, 1000)
print(crl.validacion(vc, wdbc, crl))
print(crl.validacion(vs, wdbc, crl))

crl = ClasificadorRegresionLogistica(1, 1000)
print(crl.validacion(vc, indians, crl))
print(crl.validacion(vs, indians, crl))