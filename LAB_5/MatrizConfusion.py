from EstrategiaParticionado import ValidacionSimple
import numpy as np


def espacioROC(dataset, cl, porcentaje):
    """Función para obtener la espacio ROC.
    Los valores TPR y FPR se calculan varias veces y se obtiene
    la media.

    Args:
        dataset (numpy.array): Array numpy con los datos.
        cl_constructor (Clasificador): Constructor del clasificador.
        porcentaje (int): Porcentaje destinado al test.

    Returns:
        tuple: Tupla con TPR y FPR.
    """
    vp, vn, fp, fn = matrizConfusionClasificador(dataset, cl, porcentaje)
    return TPR(vp, fn), FPR(fp, vn)


def matrizConfusionClasificador(dataset, cl, porcentaje):
    """Función para calcular la matriz de confusión
    dados unos datos y un clasificador.

    Args:
        dataset (numpy.array): Array numpy con los datos.
        cl (Clasificador): Clasificador.
        porcentaje (int): Porcentaje destinado al test.

    Returns:
        tuple: Tupla con los valores:
        - Verdaderos Positivos.
        - Verdaderos Negativos.
        - Falsos Positivos.
        - Falsos Negativos.
    """
    p = ValidacionSimple(porcentaje, 1)
    p.creaParticiones(dataset.datos)

    xTrain = dataset.datos[p.particiones[0].indicesTrain]
    xTest = dataset.datos[p.particiones[0].indicesTest]
    yTrain = dataset.datos[:, -1][p.particiones[0].indicesTest]

    cl.entrenamiento(xTrain, dataset.nominalAtributos, dataset.diccionario)
    yPred, _ = cl.clasifica(xTest, dataset.nominalAtributos, dataset.diccionario)

    return matrizConfusion(yPred, yTrain)


def matrizConfusionCluster(cluster, datos, clase):
    """Función que dado un cluster, devuelve los datos de una matríz de confusión.

    Args:
            cluster (tuple): Cluster a analizar.
            datos (numpy.array): Dataset
            clase (int): Clase mayoritaria en el cluster.

    Returns:
            tuple: Tupla con los valores:
            - Verdaderos Positivos.
            - Verdaderos Negativos.
            - Falsos Positivos.
            - Falsos Negativos.
    """
    vp = verdaderosPositivos(cluster, datos, clase)
    vn = verdaderosNegativos(cluster, datos, clase)
    fp = falsosPositivos(cluster, datos, clase)
    fn = falsosNegativos(cluster, datos, clase)

    # Si resInProb es False, lenDatos=1 y lenCluster=1.
    return vp, vn, fp, fn


def matrizConfusion(prediccion, Y):
    """Función para calcular la matriz de confusión
    dada una predicción y los datos originales.
    Esta función unicamente funciona para datos en
    los que hay 2 clases.

    Ejemplo:
            1, 0, 1, 1, 0, 1, 1 original
            0, 1, 1, 1, 0, 1, 0 prediccion
            fn fp vp vp vn vp fn

    Args:
        prediccion (list): Lista con la predicción.
        Y (numpy.array): Array numpy con la columna
        de las clases.

    Returns:
        tuple: Tupla con los valores:
        - Verdaderos Positivos.
        - Verdaderos Negativos.
        - Falsos Positivos.
        - Falsos Negativos.
    """
    vp, vn, fp, fn, = (0, 0, 0, 0)

    for clasePred, claseOrig in zip(prediccion, Y):
        if claseOrig == 1:
            if clasePred == claseOrig:
                vp += 1
            else:
                fn += 1
        else:
            if clasePred == claseOrig:
                vn += 1
            else:
                fp += 1

    return vp, vn, fp, fn


def verdaderosPositivos(cluster, datos, clase):
    """Función para calcular los verdaderos positivos.
    Obtenemos el número de datos dentro del cluster
    que pertenece a la clase mayoritaria.

    Args:
            cluster (list): Lista con los indices de los datos en el cluster.
            datos (numpy.array): Dataset.
            clase (int): Clase mayoritaria del cluster.

    Returns:
            int: Número de verdaderos positivos en el cluster.
    """
    vp = 0
    for dato in cluster:
        if int(datos[dato][-1]) == clase:
            vp += 1
    return vp


def verdaderosNegativos(cluster, datos, clase):
    """Función para calcular los verdaderos negativos.
    Obtenemos el número de datos en el dataset que:
            1. No están en el cluster
            2. La clase es distinta

    Args:
            cluster (list): Lista con los indices de los datos en el cluster.
            datos (numpy.array): Dataset.
            clase (int): Clase mayoritaria del cluster.

    Returns:
            int: Número de verdaderos negativos en el cluster.
    """
    vn = 0
    for i, fila in enumerate(datos):
        if fila[-1] != clase and i not in cluster:
            vn += 1
    return vn


def falsosPositivos(cluster, datos, clase):
    """Función para calcular los falsos positivos.
    Obtenemos el número de datos en el cluster que
    no son de la clase mayoritaria.

    Args:
            cluster (list): Lista con los indices de los datos en el cluster.
            datos (numpy.array): Dataset.
            clase (int): Clase mayoritaria del cluster.

    Returns:
            int: Número de falsos positivos en el cluster.
    """
    fp = 0
    for dato in cluster:
        if int(datos[dato][-1]) != clase:
            fp += 1
    return fp


def falsosNegativos(cluster, datos, clase):
    """Función para calcular los falsos negativos.
    Obtenemos el número de datos en el dataset que:
            1. No están en el cluster.
            2. Son de la clase mayoritaria del cluster.

    Args:
            cluster (list): Lista con los indices de los datos en el cluster.
            datos (numpy.array): Dataset.
            clase (int): Clase mayoritaria del cluster.

    Returns:
            int: Número de falsos negativos en el cluster.
    """
    fn = 0
    for i, fila in enumerate(datos):
        if fila[-1] == clase and i not in cluster:
            fn += 1
    return fn


def positivos(datos, clase):
    """Función para obtener el número de negativos

    Args:
            datos (numpy.array): Dataset.
            clase (int): Clase mayoritaria del cluster.

    Returns:
            int: Número de falsos negativos en el cluster.
    """
    p = 0
    for fila in datos:
        if fila[-1] == clase:
            p += 1
    return p


def negativos(datos, clase):
    """Función para obtener el número de negativos

    Args:
            datos (numpy.array): Dataset.
            clase (int): Clase mayoritaria del cluster.

    Returns:
            int: Número de falsos negativos en el cluster.
    """
    n = 0
    for fila in datos:
        if fila[-1] != clase:
            n += 1
    return n


def exactitud(vp, vn, fp, fn):
    """Función para calcular la exactitud dados los valores
    de la matriz de confusión.

    Args:
            vp (int): Verdaderos positivos.
            vn (int): Verdaderos negativos.
            fp (int): Falsos positivos.
            fn (int): Falsos positivos.

    Returns:
            float: Exactitud.
    """
    return (vp+vn)/(vp+fp+fn+vn)


def TPR(vp, fn):
    """Función para calcular el rate de los
    verdaderos positivos (True positive rate).

    Args:
            vp (int): Verdaderos positivos.
            fn (int): Falsos Negativos.

    Returns:
            float: TPR.
    """
    if vp+fn == 0:
        return 0
    return vp/(vp+fn)


def FNR(fn, vp):
    """Función para calcular el rate de los
    flasos negativos (false negative rate).

    Args:
            fn (int): Falsos Negativos.
            vp (int): Verdaderos positivos.

    Returns:
            float: FNR.
    """
    if vp+fn == 0:
        return 0
    return fn/(vp+fn)


def FPR(fp, vn):
    """Función para calcular el rate de los
    flasos positivos (false positive rate).

    Args:
            fp (int): Falsos positivos.
            vn (int): Verdaderos negativos.

    Returns:
            float: FPR.
    """
    if vn+fp == 0:
        return 1
    return fp/(vn+fp)


def TNR(vn, fp):
    """Función para calcular el rate de los
    verdaderos negativos (true negative rate).

    Args:
            vn (int): Verdaderos negativos.
            fp (int): Falsos positivos.

    Returns:
            float: FPR.
    """
    if fp+vn == 0:
        return 0
    return vn/(fp+vn)


def precision(vp, fp):
    """Función para calcular la precision dados los valores
    de la matriz de confusión.

    Args:
            vp (int): Verdaderos positivos.
            fp (int): Falsos positivos.

    Returns:
            float: Precision.
    """
    return vp/(vp+fp)


def sensibilidad(vp, fn):
    """Función para calcular la sensibilidad dados los valores
    de la matriz de confusión.

    Args:
            vp (int): Verdaderos positivos.
            fn (int): Falsos positivos.

    Returns:
            float: Sensibilidad.
    """
    return vp/(vp+fn)


def especificidad(vn, fp):
    """Función para calcular la especificidad dados los valores
    de la matriz de confusión.

    Args:
            vn (int): Verdaderos negativos.
            fp (int): Falsos positivos.

    Returns:
            float: especificidad.
    """
    return vn/(vn+fp)
