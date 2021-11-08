from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada, ValidacionSimple
import Clasificador
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
import utils

datos0 = Datos("ConjuntosDatos/lentillas.data")
datos1 = Datos("ConjuntosDatos/german.data")
datos2 = Datos("ConjuntosDatos/tic-tac-toe.data")

cnv0 = Clasificador.ClasificadorNaiveBayes()
cnv1 = Clasificador.ClasificadorNaiveBayes()
cnv2 = Clasificador.ClasificadorNaiveBayes()
cnv3= Clasificador.ClasificadorNaiveBayes()

particionado1 = ValidacionCruzada(4)
particionado2 = ValidacionSimple(25, 4)

cnv3.entrenamiento(datos0.datos, datos0.nominalAtributos, datos0.diccionario)


print("Lentillas, particionado cruzado: ", cnv0.validacion(particionado1, datos0, cnv0))
print("Lentillas, particionado simple: ", cnv0.validacion(particionado2, datos0, cnv0))
cnv3 = Clasificador.ClasificadorNaiveBayes()
gnb = GaussianNB()
mnb = MultinomialNB(fit_prior=True)
mnb2 = MultinomialNB(fit_prior=True, alpha=1)
X = datos0.datos[:,[i for i in range(datos0.datos.shape[1]-1)]]
y = datos0.datos[:,-1]
gnb.fit(X, y)
mnb.fit(X, y)
mnb2.fit(X, y)
print("Gaussian:", cross_val_score(gnb, X, y, cv=4).mean())
print("Multinomial:", cross_val_score(mnb, X, y, cv=4).mean())
print("Multinomial CLP:", cross_val_score(mnb2, X, y, cv=4).mean())
cnv3.entrenamiento(datos0.datos, datos0.nominalAtributos, datos0.diccionario)
a, b = cnv3.clasifica(datos0.datos, datos0.nominalAtributos, datos0.diccionario)
print(1-cnv3.error(datos0.datos, a))
print(1-cnv3.error(datos0.datos, b))
print()

print("German, particionado cruzado: ", cnv1.validacion(particionado1, datos1, cnv1))
print("German, particionado simple: ", cnv1.validacion(particionado2, datos1, cnv1))
cnv3 = Clasificador.ClasificadorNaiveBayes()
gnb = GaussianNB()
mnb = MultinomialNB(fit_prior=True, alpha=1)
mnb2 = MultinomialNB(fit_prior=True, alpha=0)
cnb = CategoricalNB()
X = datos1.datos[:,[i for i in range(datos1.datos.shape[1]-1)]]
y = datos1.datos[:,-1]
gnb.fit(X, y)
mnb.fit(X, y)
mnb2.fit(X, y)
cnb.fit(X, y)
print("Gaussian: ", cross_val_score(gnb, X, y, cv=4).mean())
print("Multinomial CLP: ", cross_val_score(mnb, X, y, cv=4).mean())
print("Multinomial: ", cross_val_score(mnb2, X, y, cv=4).mean())
#print("Categorical: ", cross_val_score(cnb, X, y, cv=4).mean())
cnv3.entrenamiento(datos1.datos, datos1.nominalAtributos, datos1.diccionario)
a, b = cnv3.clasifica(datos1.datos, datos1.nominalAtributos, datos1.diccionario)
print(1-cnv3.error(datos1.datos, a))
print(1-cnv3.error(datos1.datos, b))
print()

print("tic-tac-toe, particionado cruzado: ", cnv2.validacion(particionado1, datos2, cnv2))
print("tic-tac-toe, particionado simple: ", cnv2.validacion(particionado2, datos2, cnv2))
cnv3 = Clasificador.ClasificadorNaiveBayes()
gnb = GaussianNB()
mnb = MultinomialNB()
cnb = CategoricalNB()
X = datos2.datos[:,[i for i in range(datos2.datos.shape[1]-1)]]
y = datos2.datos[:,-1]
gnb.fit(X, y)
mnb.fit(X, y)
cnb.fit(X, y)
print("Gaussian:", cross_val_score(gnb, X, y, cv=4).mean())
print("Multinomial:", cross_val_score(mnb, X, y, cv=4).mean())
#print("Categorical:", cross_val_score(cnb, X, y, cv=4).mean())
cnv3.entrenamiento(datos2.datos, datos2.nominalAtributos, datos2.diccionario)
a, b = cnv3.clasifica(datos2.datos, datos2.nominalAtributos, datos2.diccionario)
print(1-cnv3.error(datos2.datos, a))
print()
