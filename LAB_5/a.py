from ClasificadorAlgoritmoGenetico import AlgoritmoGenetico
from Datos import Datos
import utils

lentillas = Datos("lentillas.data")
datos = Datos("tic-tac-toe.data")
datos = Datos("titanic.csv", allNominal=True)

errorMediolen1_inter_estandar, errorMediolen2_inter_estandar = utils.AG_test(lentillas, lentillas, AlgoritmoGenetico.cruceInterReglas, AlgoritmoGenetico.mutacionEstandar)
errorMediolen1_intra_estandar, errorMediolen2_intra_estandar = utils.AG_test(lentillas, lentillas, AlgoritmoGenetico.cruceIntraReglas, AlgoritmoGenetico.mutacionEstandar)
errorMediolen1_inter_reglas, errorMediolen2_inter_reglas = utils.AG_test(lentillas, lentillas, AlgoritmoGenetico.cruceInterReglas, AlgoritmoGenetico.mutacionReglas)
errorMediolen1_intra_reglas, errorMediolen2_intra_reglas = utils.AG_test(lentillas, lentillas, AlgoritmoGenetico.cruceIntraReglas, AlgoritmoGenetico.mutacionReglas)
utils.plot_comp([errorMediolen1_inter_estandar, errorMediolen1_inter_reglas], [errorMediolen1_intra_estandar, errorMediolen1_intra_reglas], ["Cruce Inter", "Cruce Intra"])
ag = AlgoritmoGenetico(30, 50, 5, AlgoritmoGenetico.cruceInterReglas, AlgoritmoGenetico.mutacionReglas, 0.2, 0.05, show=True)
ag.entrenamiento(datos.datos, datos.nominalAtributos, datos.diccionario)
print(ag.reglasMejor(datos.diccionario))