from ClasificadorAlgoritmoGenetico import AlgoritmoGenetico
from Clasificador import ClasificadorNaiveBayes
from EstrategiaParticionado import ValidacionSimple
import MatrizConfusion as mc
import matplotlib.pyplot as plt

porcentajesTest = [20]
poblaciones = [50, 150]
generaciones = [100, 200]
MAXREGLAS = 5

def AG_test(ttt, titanic, cruce, mutacion):
    errorMedioTTT = { poblacion: { generacion: [] for generacion in generaciones } for poblacion in poblaciones }
    errorMedioTITANIC = { poblacion: { generacion: [] for generacion in generaciones } for poblacion in poblaciones }

    for poblacion in poblaciones:
        for generacion in generaciones:
            for porcentaje in porcentajesTest:
                VS = ValidacionSimple(porcentaje, 1)
                mediaTTT, TTT_reglas_str = AG_test_generacion_poblaciones(cruce, mutacion, poblacion, generacion, ttt, VS)
                mediaTITANIC, TITANIC_reglas_str = AG_test_generacion_poblaciones(cruce, mutacion, poblacion, generacion, titanic, VS)
                errorMedioTTT[poblacion][generacion].append(mediaTTT)
                errorMedioTITANIC[poblacion][generacion].append(mediaTITANIC)
                print(f"Test Poblacion={poblacion} Generaciones={generacion}\tTic-Tac-Toe - Error={mediaTTT:2f}\t\tTitanic - Error={mediaTITANIC:2f}")
                print("\n[Tic-Tac-Toe] reglas del mejor individuo:\n", TTT_reglas_str)
                print("[Titanic] reglas del mejor individuo:\n", TITANIC_reglas_str)

    return errorMedioTTT, errorMedioTITANIC


def AG_test_generacion_poblaciones(cruce, mutacion, generacion, poblacion, dataset, particionado):
    ag = AlgoritmoGenetico(poblacion, generacion, MAXREGLAS, cruce, mutacion, 0.05, 0.05)
    r = ag.validacion(particionado, dataset, ag)[0]
    reglas_str = ag.reglasMejor(dataset.diccionario)
    return r, reglas_str


def plot(data, nombreData, cruce, mutacion):
    string = "Evolución del fitness "+nombreData
    X = [i+1 for i in range(100)]
    medias = []
    fitnesses = []
    for _ in range(5):
        fitness_data = []
        ag = AlgoritmoGenetico(20, 100, MAXREGLAS, AlgoritmoGenetico.cruceInterReglas, AlgoritmoGenetico.mutacionReglas, 0.05, 0.05)
        ag.entrenamiento(data.datos, data.nominalAtributos, data.diccionario, fitnessData=fitness_data)
        fitnesses.append([f for f, _ in fitness_data])
        medias.append([m for _, m in fitness_data])

    plt.figure(figsize=(20,20))
    plt.subplots()
    for _, f in enumerate(fitnesses):
        plt.plot(X, f)    
        plt.xlabel("Generaciones")
        plt.ylabel("Fitness")
    plt.title(string)

    plt.subplots()
    for _, m in enumerate(medias):
        plt.plot(X, m)    
        plt.title(string)
        plt.xlabel("Generaciones")
        plt.ylabel("Fitness")
    plt.title("Fitness medio en "+nombreData)

def espacio_ROC_avg_AG(dataset, times, cruce, mutacion, porcentaje):
    tpr_media = []
    fpr_media = []
    for _ in range(times):
        ag = AlgoritmoGenetico(150, 200, MAXREGLAS, cruce, mutacion, 0.05, 0.05)
        tpr, fpr = mc.espacioROC(dataset, ag, porcentaje)
        tpr_media.append(tpr)
        fpr_media.append(fpr)
    return sum(tpr_media)/len(tpr_media), sum(fpr_media)/len(fpr_media)


def espacio_ROC_avg_NB(dataset, times, porcentaje):
    tpr_media = []
    fpr_media = []
    for _ in range(times):
        nb = ClasificadorNaiveBayes()
        tpr, fpr = mc.espacioROC(dataset, nb, porcentaje)
        tpr_media.append(tpr)
        fpr_media.append(fpr)
    return sum(tpr_media)/len(tpr_media), sum(fpr_media)/len(fpr_media)


def plot_espacio_ROC(dataset, times, porcentaje, cruce, mutacion):
    X_NB, Y_NB = espacio_ROC_avg_NB(dataset, times, porcentaje)
    X_AG, Y_AG = espacio_ROC_avg_AG(dataset, times, cruce, mutacion, porcentaje)
    X = [X_NB, X_AG]
    Y = [Y_NB, Y_AG]
    labels = ["Naive Bayes", "Algoritmo Genético"]

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
