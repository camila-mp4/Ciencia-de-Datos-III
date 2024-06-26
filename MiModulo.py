import random

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.metrics import auc
from scipy.stats import chi2
from scipy.stats import norm

class ResumenNumerico():
  """Clase que realiza el resumen numérico de una muestra de una variable
  cuantitativa. Calcula media, desvío, cuartiles, mínimo y máximo de la misma.
  Args:
      datos: vector con las observaciones de la muestra."""
  def __init__(self, datos):
    """Inicializa instancia de la clase ResumenNumerico."""
    self.datos = datos

  def __str__(self):
    """Retorno de aplicar print() a un objeto de la clase ResumenNumerico."""
    resumen = f"""\033[1mResumen numérico de los datos. \033[0m
Media: {self.media()}
Desvío estándar: {round(self.desvio(),3)}
Mínimo: {self.min()}
Máximo: {self.max()}
Q1 = {self.cuartiles()[0]}
Q2/Mediana = {self.mediana()}
Q3 = {self.cuartiles()[1]}"""
    return resumen

  def media(self) -> float:
    """Calcula media de la muestra."""
    return np.mean(self.datos)

  def desvio(self) -> float:
    """Calcula desvío de la muestra."""
    return np.std(self.datos)

  def cuartiles(self):
    """Halla el cuartil 1 y el cuartil 3 de la muestra.
    Para cuartil 2 ver el método mediana()"""
    q1, q3 = np.quantile(self.datos, 0.25), np.quantile(self.datos, 0.75)
    return [q1, q3]

  def mediana(self):
    """Devuelve la mediana de la muestra."""
    return np.median(self.datos)

  def min(self):
    """Devuelve el mínimo de la muestra."""
    return min(self.datos)

  def max(self):
    """Devuelve el máximo de la muestra."""
    return max(self.datos)

  def resumen(self) -> dict:
    """Genera y devuelve diccionario con media, desvío, mínimo, máximo y
    cuartiles de la muestra."""
    diccionario_resumen = {
        "Media" : self.media(),
        "Desvío estándar" : self.desvio(),
        "Mínimo" : self.min(),
        "Máximo" : self.max(),
        "Q1" : self.cuartiles()[0],
        "Q3" : self.cuartiles()[1],
        "Mediana" : self.mediana(),
     }
    return diccionario_resumen

class ResumenGrafico():
  """Realiza distintas gráficas que permiten visualizar la distribución estimada
  de una muestra de datos cuantitativa.

  Contiene métodos para estimar la distribución mediante kernel uniforme,
  cuadrático, triangular y normal. Permite realizar qqplots de los datos contra
  cuantiles de la distribución normal estándar y exponencial.

  Args:
      datos: vector de observaciones de una variable cuantitativa.
  """

  def __init__(self, datos):
    """Inicializa instancia de la clase ResumenGrafico."""
    self.datos = np.array(datos)

  def histograma(self,h: float) -> tuple:
    """Define la función histograma asociada a la muestra de datos ingresada.

    Args:
        h (float): ancho de los bins del histograma.

    Returns:
        bins (np.ndarray): intervalos de ancho h en la muestra observada.
        densidad (np.ndarray): frecuencia relativa de los datos observados en
        cada intervalo definido por los bins.
    """
    # generamos vector con extremos de cada intervalo/bin
    bins = np.arange(min(self.datos),max(self.datos), h)
    # generamos vector para almacenar frecuencia de datos en cada bin
    frecuencias = np.zeros(len(bins))

    for i in range(len(bins)): # recorremos bins
      if i+1 == len(bins): # si ya recorrimos todos finalizamos el proceso
        break
      # hallamos indices de datos mayores al extremo izquierdo
      ind_m = np.where(self.datos >= bins[i])
      # hallamos indices de datos menores al extremo derecho
      ind_M = np.where(self.datos < bins[i+1])
      # intsercamos para dejar solo indices de datos entre ambos extremos
      ind_bins = np.intersect1d(ind_m, ind_M)

      frecuencias[i] = len(ind_bins) # long de vector de indices = cant de datos en el intervalo

    densidad = frecuencias / (len(self.datos) * h) # hallamos densidad de los datos

    return bins, densidad

  def evaluacion_histograma(self,h:float,x:np.ndarray) -> np.ndarray:
    """Evalúa el histograma en un vector ordenado x.

    Args:
        h (float): ancho de los bins del histograma.
        x (np.ndarray): vector en el que se evaluará el histograma.

    Returns:
        evaluacion_histo (np.ndarray): H(x_i) para cada x_i en el vector x, donde
        H(x) es la función histograma asociada a los datos.
    """
    evaluacion_histo = np.zeros(len(x))
    intervalos, valor = self.histograma(h)

    for i in range(len(intervalos)):
      if i+1 == len(intervalos):
        break

      x_mayores = np.where(intervalos[i] <= x)
      x_menores = np.where(x < intervalos[i+1])
      indices = np.intersect1d(x_mayores, x_menores)

      evaluacion_histo[indices] = valor[i]

    return evaluacion_histo

  def kernel_uniforme(self, h:float, x: np.ndarray) -> np.ndarray:
    """Estima la densidad de los datos en los puntos del vector x con kernel
    uniforme y ancho de ventana h.

    Args:
        h (float): ancho de los bins del histograma.
        x (np.ndarray): vector en el que se evaluará el histograma.

    Returns:
        densidad (np.ndarray): densidad estimada para cada punto de x.
    """
    densidad = np.zeros(len(x))

    for i in range(len(x)):
      u = (self.datos - x[i]) / h

      densidad[i] = (np.sum(u <= 1/2) - np.sum(u <= -1/2))

    return densidad

  def kernel_gaussiano(self, h:float, x: np.ndarray) -> np.ndarray:
    """Estima la densidad de los datos en los puntos del vector x con kernel
    gaussiano y ancho de ventana h.

    Args:
        h (float): ancho de los bins del histograma.
        x (np.ndarray): vector en el que se evaluará el histograma.

    Returns:
        densidad (np.ndarray): densidad estimada para cada punto de x.
    """
    data = self.datos.copy()
    densidad = np.zeros(len(x))
    for i in range(len(x)):
      u = (data - x[i]) / h
      densidad[i] = np.sum((np.e ** ((-1/2) * u**2)) / np.sqrt(2 * np.pi))

    return densidad

  def kernel_cuadratico(self, h:float, x: np.ndarray) -> np.ndarray:
    """Estima la densidad de los datos en los puntos del vector x con kernel
    cuadrático y ancho de ventana h.

    Args:
        h (float): ancho de los bins del histograma.
        x (np.ndarray): vector en el que se evaluará el histograma.

    Returns:
        densidad (np.ndarray): densidad estimada para cada punto de x.
    """
    data = self.datos.copy()
    densidad = np.zeros(len(x))
    for i in range(len(x)):
      u = (data - x[i]) / h
      mayores = np.where(-1 <= u)
      menores = np.where(u <= 1)
      indices = np.intersect1d(mayores,menores)
      u_intervalo = u[indices]

      densidad[i] = np.sum( 3/4 * (1 - u_intervalo**2))

    return densidad

  def kernel_triangular(self, h:float, x: np.ndarray) -> np.ndarray:
    """Estima la densidad de los datos en los puntos del vector x con kernel
    triangular y ancho de ventana h.

    Args:
        h (float): ancho de los bins del histograma.
        x (np.ndarray): vector en el que se evaluará el histograma.

    Returns:
        densidad (np.ndarray): densidad estimada para cada punto de x.
    """
    data = self.datos.copy()
    densidad = np.zeros(len(x))

    for i in range(len(x)):
      u = (data - x[i]) / h

      mayores_menos1 = np.where(-1 <= u)
      menores_0 = np.where(u <= 0)

      mayores_0 = np.where(0 <= u)
      menores_1 = np.where(u <= 1)

      entre_menos1_0 = np.intersect1d(mayores_menos1, menores_0)
      entre_0_1 = np.intersect1d(mayores_0, menores_1)

      densidad[i] = np.sum((1 + u[entre_menos1_0])) + np.sum((1 - u[entre_0_1]))

    return densidad

  def mi_densidad(self, h: float, x: np.ndarray, kernel: str) -> np.ndarray:
    """Estima la densidad de los datos en los puntos del vector x con el kernel
    elegido y ancho de ventana h.

    Args:
        h (float): ancho de los bins del histograma.
        x (np.ndarray): vector en el que se evaluará el histograma.
        kernel (string): kernel a utilizar.
        kernel (str): kernel a utilizar para la estimación. Puede ser uniforme,
        gaussiano, cuadrático o triangular.

    Returns:
        densidad (np.ndarray): densidad estimada para cada punto de x con el
        kernel especificado y ancho de ventana h.
    """
    density = np.zeros(len(x))
    if kernel == 'uniforme':
      density = self.kernel_uniforme(h,x)

    if kernel == 'gaussiano':
      density = self.kernel_gaussiano(h,x)

    if kernel == 'cuadrático':
      density = self.kernel_cuadratico(h,x)

    if kernel == 'triangular':
      density = self.kernel_triangular(h,x)

    density = density / (len(self.datos) * h)
    return density

  def miqqplot(self):
    """Realiza qq-plot normal para los cuantiles de los datos observados."""

    # ordenamos y normalizamos los datos
    data_ord = np.sort(self.datos)
    media = np.mean(self.datos)
    desvio = np.std(self.datos)

    data_ord_s = (data_ord - media) / desvio

    # generamos el vector p
    n = len(self.datos)

    p = np.arange(1, n+1) / (n+1)

    # calculamos los cuantiles de los datos y teóricos
    cuantiles_muestrales = np.zeros(len(p))
    cuantiles_teoricos = np.zeros(len(p))

    for i in range(len(p)):
      cuantiles_teoricos[i] = norm.ppf(p[i])
      cuantiles_muestrales[i] = np.quantile(data_ord_s, p[i])

    plt.scatter(cuantiles_teoricos, cuantiles_muestrales, color='blue', marker='o')
    plt.xlabel('Cuantiles teóricos')
    plt.ylabel('Cuantiles muestrales')
    plt.plot(cuantiles_teoricos,cuantiles_teoricos , linestyle='-', color='red')
    plt.show()

class TrainTest():
  """Clase que separa conjunto de datos en conjuntos para entrenar y testear modelos.

  Args:
      data (pd.Dataframe): conjunto de datos a separar
      p (float): porcentaje de datos para conjunto de entrenamiento. 0 < p < 1.
      semilla (int): opcional, permite definir la semilla de selección de datos.
  """
  def __init__(self, data: pd.DataFrame, p:float, semilla:int = 10):
    """Inicializa instancia de la clase TrainTest."""
    random.seed(semilla)

    self.indices = random.sample(range(len(data)), int(len(data) * p))

    self.train = data.iloc[self.indices]
    self.test = data.drop(self.indices)

  def datos_train(self) -> pd.DataFrame:
    """Devuelve conjunto de entrenamiento."""
    return self.train

  def datos_test(self) -> pd.DataFrame:
    """Devuelve conjunto de testeo."""
    return self.test

class Regresion():
  """Realiza cálculos de regresión con librería statmodels.

    Atributos:
        x (np.ndarray): vector de observaciones de la variable predictora.
        y (np.ndarray): vector de observaciones de la variable respuesta.
        X (np.ndarray): matriz de diseño asociada a x."""

  def __init__(self, predictoras: np.ndarray, respuesta: np.ndarray):
    """Inicializa una instancia de la clase regresión."""
    self.x = predictoras
    self.y = respuesta
    self.X = sm.add_constant(self.x)

  def valores_ajustados(self) -> np.ndarray:
    """Devuelve los valores ajustados del modelo de regresión."""
    return self.resultado.fittedvalues

  def betas(self) -> np.ndarray:
    """Devuelve las estimaciones de los parámetros asociados a cada predictora."""
    return self.resultado.params

  def se(self) -> np.ndarray:
    """Devuelve los errores estándar asociados a cada beta estimado."""
    return self.resultado.bse

  def t_obs(self) -> np.ndarray:
    """Devuelve el estadístico t observado para cada beta estimado."""
    return self.resultado.tvalues

  def pvalues(self) -> np.ndarray:
    """Devuelve el p-valor asociado a cada beta estimado."""
    return self.resultado.pvalues

  def design_matrix(self,vector: np.ndarray) -> np.ndarray:
    """Devuelve matriz de diseño para el vector de observaciones ingresado.
    Args:
        vector (np.ndarray): vector con una o más observaciones de una o más
        variables predictoras.
    """
    if vector.ndim == 0 or vector.ndim == 1: # si es una única observación de una o varias predictoras
      matriz_diseño = np.insert(vector, 0, 1)
    else: # si hay más de una observacion
      matriz_diseño = sm.add_constant(vector, has_constant='add')
    return matriz_diseño

class RegresionLineal(Regresion):
  """Esta clase es una instancia de la clase Regresion y realiza cálculos
  y gráficos para modelos de regresión lineal utilizando las librerías numpy,
  statsmodels y matplotlib.
  """

  def __init__(self, predictoras: np.ndarray, respuesta: np.ndarray):
    """Inicializa una instancia de la clase Regresión Lineal con la librería
    statsmodels.

    Atributos:
        modelo: instancia de la clase statmodels.OLS
        resultado: instancia de la clase statmodels.Results
    """
    super().__init__(predictoras, respuesta)
    self.modelo = sm.OLS(self.y, self.X)
    self.resultado = self.modelo.fit()

  def coeficientes_correlacion(self):
    """Devuelve los coeficientes de correlación entre cada variable predictora
    y la variable respuesta.

    Returns:
        np.ndarray / float: Coeficientes de correlación entre x e y.
    """

    if self.x.ndim == 1: # si tenemos una sola predictora
      return np.corrcoef(self.x, self.y, rowvar = False)[1,0] # corrcoef para (x_1, y)
    else:
      return np.corrcoef(self.x, self.y, rowvar = False)[len(self.x.T), :len(self.x.T)] # corrcoef para (x_i, y)

  def r_cuadrado(self) -> float:
    """Devuelve el coeficiente R cuadrado del modelo ajustado."""
    return self.resultado.rsquared

  def r_cuadrado_ajustado(self) -> float:
    """Devuelve el coeficiente R cuadrado ajustado del modelo ajustado."""
    return self.resultado.rsquared_adj

  def grafico_dispersion(self):
    """Realiza un gráfico de dispersión para cada variable predictora.
    En caso de haber una sola, superpone la recta ajustada."""

    cant_predictoras = self.x.T.shape[0]

    if self.x.ndim == 1: # si tenemos una sola predictora
      plt.figure()
      plt.title(f"Variable respuesta en función de la predictora.")
      plt.scatter(self.x, self.y)
      plt.plot(self.x, self.valores_ajustados())

    else: # si tenemos más de una predictora
              # realizamos un gráfico por predictora
      for i in range(cant_predictoras):
        plt.figure()
        plt.title(f"Variable respuesta en función de X_{i}")
        plt.scatter(self.x[:,i], self.y)
        plt.plot(self.x, self.valores_ajustados())
        plt.show()

  def predecir_valor(self, x_new: np.ndarray):
    """Predice el valor  de la variable respuesta asociado a nuevos valores
    de las variables predictoras.
      Args:
            x_new: vector de variables predictoras.
      Returns:
            np.ndarray: predicciones.
    """
    X_new = self.design_matrix(x_new)
    return X_new @ self.betas()

  def ECM(self, x_test: np.ndarray, y_test:np.ndarray) -> float:
    """Devuelve el error cuadrático medio asociado a una muestra de testeo.
    Args:
        x_test (np.ndarray): predictoras para testear.
        y_test (np.ndarray): valores de la variable respuesta para testear.

    Returns:
        error_cuadratico_medio (float): estimación del ECM del modelo.
    """
    error_cuadratico_medio = np.sum((y_test - self.predecir_valor(x_test))**2) / len(self.y)
    return error_cuadratico_medio

  def intervalos(self, x_new, alfa=0.05, mostrar = True):
    """Devuelve intervalos de confianza y de predicción para un valor independiente
    de las variables predictoras.
      Args:
          x_new (np.ndarray): vector con valores independientes de las variables
          predictoras.
          alfa (float): número entre 0 y 1 para definir significancia del test.
          mostrar (bool): si True imprime el intervalo, si no sólo lo devuelve.
    """
    X_new = np.insert(x_new, 0, 1)
    prediccion = self.resultado.get_prediction(X_new)

    int_confianza = prediccion.conf_int(alpha = alfa)
    int_prediccion = prediccion.conf_int(obs = True, alpha = alfa)

    if mostrar == True:
      print(f"El intervalo de confianza para mu_(Y| {x_new}) es: {int_confianza[0]}")
      print(f"El intervalo de predicción para X = x_new es: {int_prediccion[0]}")

    return int_confianza[0], int_prediccion[0]

  def analisis_residuos(self):
    """Imprime qqplot normal de los residuos del modelo y un gráfico de
    dispersión de los mismos versus los valores predichos."""
    residuos = self.resultado.resid
    plt.title("Residuos vs. valores predichos.")
    plt.scatter(self.valores_ajustados(), self.resultado.resid)
    plt.show()

    residuos_normalizados = (np.sort(residuos) - np.mean(residuos)) / np.std(residuos)

    sm.qqplot(residuos_normalizados, line='45')
    plt.title('QQplot normal de los residuos')

class RegresionLogistica(Regresion):
  """Esta clase es una instancia de la clase Regresion y realiza cálculos
  y gráficos para modelos de regresión logística utilizando las librerías numpy,
  statsmodels y matplotlib.
  """

  def __init__(self, predictoras, respuesta):
    super().__init__(predictoras, respuesta)
    self.modelo = sm.Logit(self.y, self.X);
    self.resultado = self.modelo.fit();
    self.probabilidades_estimadas = (np.e ** self.valores_ajustados()) / (np.e ** self.valores_ajustados() + 1)

  def estimar_probabilidad(self, x_new: np.ndarray) -> np.ndarray:
    """Dado un vector de prueba x_new, estima la probabilidad de que la variable
    respuesta tome tal valor con el modelo ajustado.

    Args:
        x_test (np.ndarray): vector de valores independientes de las predictoras.
    """

    # creamos matriz de diseño
    X_new = self.design_matrix(x_new)

    # multiplicamos por los parámetros estimados para hallar exponentes
    exp = X_new @ self.betas()

    # evaluamos exponentes en la función logística
    probabilidades = np.e ** exp / (1 + np.e ** exp)
    return probabilidades

  def categoriza(self, x_test: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Dado un umbral p y un vector de prueba x_test, categoriza como 1 aquellas
    probabilidades mayores a p y 0 a las demás.
    Args:
        x_test (np.ndarray): vector de valores independientes de las predictoras.
        p (float): punto de corte para clasificar.
    """
    # asignamos valor 1 a las probabilidades mayor al umbral elegido, 0 a las demás
    categorizadas = 1 * (self.estimar_probabilidad(x_test) >= p)
    return categorizadas

  def matriz_confusion(self, x_test: np.ndarray, y_test: np.ndarray, p: float = 0.5, mostrar: bool = True) -> dict:
    """Calcula matriz de confusión del modelo ajustado para los datos de prueba
    x_test y el p ingresado.

    Args:
        x_test (np.ndarray): valores independientes de las predictoras.
        y_test (np.ndarray): valores de la variable respuesta observados para
        cada valor de las predictoras en x_test.
        p (float): punto de corte para clasificar.
        mostrar (bool): define si imprimir la matriz o no.

    Returns:
        resultados (dict): diccionario con cada entrada de la matriz, especificidad
        y sensibilidad del modelo.
    """
    # categorizamos los valores predichos para x_test
    y_pred = self.categoriza(x_test, p)

    # comparamos y_pred con y_test para hallar matriz de confusión
    a = np.sum((y_pred == 1) & (y_test == 1)) # verdaderos positivos
    b = np.sum((y_pred == 1) & (y_test == 0)) # falsos positivos
    c = np.sum((y_pred == 0) & (y_test == 1)) # falsos negativos
    d = np.sum((y_pred == 0) & (y_test == 0)) # verdaderos negativos

    tabla =  pd.DataFrame({
    'y_test = 1': [a, c],
    'y_test = 0': [b, d],
    }, index=['y_pred = 1', 'y_pred = 0']) # creamos matriz como DataFrame

    # calculamos error de mala clasificación, sensibilidad y especificidad
    error_mala_clasificacion = (b + c) / len(y_pred)
    sensibilidad = a / (a + c) # prop. de positivos detectados
    especificidad = d / (b + d) # prop. de negativos detectados

    if mostrar == True:
      print()
      print(tabla)
      print()
      print("El error de mala clasificación es", error_mala_clasificacion)
      print()
      print("La sensibilidad del modelo es ", sensibilidad)
      print("La especificidad del modelo es ", especificidad)

    resultados = {'Verdaderos positivos': a,
                  'Falsos positivos': b,
                  'Falsos negativos': c,
                  'Verdaderos negativos': d,
                  'sensibilidad': sensibilidad,
                  'especificidad': especificidad}

    return resultados

  def ROC_aux(self, x_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """Calcula la sensibilidad y especificidad del modelo para distintos
    umbrales de clasificación.

    Args:
        x_test (np.ndarray): valores independientes de las predictoras.
        y_test (np.ndarray): valores de la variable respuesta observados para
        cada valor de las predictoras en x_test.

    Returns: tupla con vector de sensibilidad y especificidad.
    """
    p = np.linspace(0, 1, 100) # vector de puntos de corte equiespaciados

    sensibilidad = np.zeros(len(p)) # sensibilidad para cada valor de p
    especificidad = np.zeros(len(p)) # especificidad para cada valor de p

    for i in range(len(p)): # para cada umbral p hallamos sensibilidad y especificidad
      sensibilidad[i] = self.matriz_confusion(x_test, y_test, p[i], mostrar = False)['sensibilidad']
      especificidad[i] = self.matriz_confusion(x_test, y_test, p[i], mostrar = False)['especificidad']

    return sensibilidad, especificidad

  def ROC(self, x_test: np.ndarray, y_test: np.ndarray, mostrar = True) -> float:
    """Imprime curva ROC del modelo y devuelve los vectores '1 - especificdad'
    y 'sensibilidad'.

    Args:
        x_test (np.ndarray): valores independientes de las predictoras.
        y_test (np.ndarray): valores de la variable respuesta observados para
        cada valor de las predictoras en x_test.

    Returns:
        ejes (tuple): tupla con eje x (1 - especificidad) e y (sensibilidad) de
        la curva ROC.
    """
    sensibilidad, especificidad = self.ROC_aux(x_test, y_test)
    x, y = 1 - especificidad, sensibilidad
    if mostrar == True:
      plt.plot(x, y)

    return x, y

  def indice_youden(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
    """Calcula el punto de corte en el que la sensibilidad y la especificidad
    del modelo son las mejores.
    Args:
        x_test (np.ndarray): valores independientes de las predictoras.
        y_test (np.ndarray): valores de la variable respuesta observados para
        cada valor de las predictoras en x_test.
    Returns:
        p[ind_j] (float): p que maximiza sensibilidad y especificidad.
    """
    p = np.linspace(0, 1, 100)

    sensibilidad, especificidad = self.ROC_aux(x_test, y_test)
    formula = sensibilidad + especificidad - 1

    # nos interesa el p en el cual se maximiza la fórmula anterior
    ind_j = np.where(formula == max(formula))

    return p[ind_j]

  def AUC_modelo(self,  x_test: np.ndarray, y_test: np.ndarray) -> float:
    """Devuelve el área bajo la curva ROC asociada al modelo ajustado.

    Args:
        x_test (np.ndarray): valores independientes de las predictoras.
        y_test (np.ndarray): valores de la variable respuesta observados para
        cada valor de las predictoras en x_test.

    Returns:
        auc (float): área bajo la curva ROC.
    """
    x, y = self.ROC(x_test, y_test, False)
    return auc(x, y)

  def clasifica_modelo_AUC(self, x_test: np.ndarray, y_test: np.ndarray) -> str:
    """Clasifica el modelo ajustado según el área bajo la curva ROC.

    Args:
        x_test (np.ndarray): valores independientes de las predictoras.
        y_test (np.ndarray): valores de la variable respuesta observados para
        cada valor de las predictoras en x_test.

    Returns:
        clasificacion (string): cadena con la clasificación asignada al modelo.
    """

    AUC = self.AUC_modelo(x_test, y_test) # hallamos AUC para clasificar el modelo

    if AUC <= 0.6: # utilizamos la clasificación propuesta en clase
      clasificacion = 'Modelo fallido.'
    elif 0.6 < AUC <= 0.7:
      clasificacion = 'Modelo pobre.'
    elif 0.7 < AUC <= 0.8:
      clasificacion = 'Modelo regular.'
    elif 0.8 < AUC <= 0.9:
      clasificacion = 'Modelo bueno.'
    else:
      clasificacion = 'Modelo excelente.'

    return clasificacion

class test_chi2():
  """Clase que realiza test chi^2 para un vector de probabilidades esperado
  pi_0 y el vector de probabilidades observado en una muestra.

  Args:
      pi_0 (np.ndarray): vector de probabilidades esperadas para los datos.
      muestra (np.ndarray): vector de observaciones o de frecuencias observadas.
      son_frecuencias (bool): True si el vector ingresado es de frecuencias
      absolutas observadas, False si es de observaciones (es decir, si registra
      a que categoría pertenece cada observación registrada.)
  """
  def __init__(self, pi_0: np.ndarray, muestra:np.ndarray, son_frecuencias = True):
    self.pi_0 = pi_0
    self.son_frecuencias = son_frecuencias
    self.muestra = muestra
    self.frec_obs = None

  def frecuencias_observadas(self) -> np.ndarray:
    """Calcula frecuencias observadas de la muestra."""
    if self.son_frecuencias: # si la clase se instanció con frecuencias observadas
      frecuencias =  self.muestra # no es necesario hacer nada

    else: # si se ingresó un vector de observaciones
      # hallamos todas las categorías observadas
      categorias = list(set(self.muestra))

      # creamos vector para almacenar frecuencia absoluta de cada categoría
      frecuencias = np.zeros(len(categorias))

      for i in range(len(categorias)): # recorremos categorías
        # almacenamos en la posición i la frecuencia de la categoría i
        frecuencias[i] = sum(self.muestra == categorias[i])

    self.frec_obs = frecuencias

    return frecuencias

  def n(self) -> int:
    """Devuelve total de observaciones de la muestra."""
    if self.son_frecuencias:
      n = sum(self.frecuencias_observadas())
    else:
      n = len(self.muestra)
    return n

  def k(self) -> int:
    """Devuelve total de categorías involucradas."""
    if self.son_frecuencias:
      k = len(self.muestra) # contamos cant. de frecuencias observadas
    else:
      k = len(list(set(self.muestra))) # contamos cant. de observaciones distintas
    return k

  def frecuencias_esperadas(self) -> np.ndarray:
    """Devuelve frecuencias esperadas para cada categoría según vector pi_0
    ingresado y total de observaciones."""
    return self.pi_0 * self.n()

  def estadistico_observado(self) -> float:
    """Devuelve estadístico observado para el test.
    """
    estadistico = np.sum((self.frecuencias_observadas() - self.frecuencias_esperadas())**2 / self.frecuencias_esperadas())
    return estadistico

  def region_rechazo(self, alfa: float = 0.05) -> float:
    """Halla el percentil alfa de la distribución chi cuadrado con k grados de
    libertad.
    Args:
        alfa (float): número entre 0 y 1 que define significancia del test."""
    df = self.k() - 1 # cantidad de categorías - 1
    percentil = chi2.ppf(1 - alfa, df)
    return percentil

  def pvalor(self) -> float:
    """Halla el pvalor del estadístico chi^2 observado."""
    df = self.k() - 1
    observado = self.estadistico_observado()
    return 1 - chi2.cdf(observado, df)

  def test(self, alfa = 0.05) -> bool:
    """Realiza test de hipótesis para H_0: pi_0 = frec_obs vs H_1: pi_0 != frec_obs.
    Args:
        alfa (float): valor entre 0 y 1 que determina significancia del test.

    Returns:
        hipotesis_nula (bool): conclusión obtenida sobre la hipótesis nula."""
    estadistico = self.estadistico_observado()
    percentil = self.region_rechazo()
    pvalor = self.pvalor()

    print(f"El estadístico observado por el test es {estadistico}.")
    print(f"Su pvalor es {pvalor}.")
    print(f"La región de rechazo queda definida por los valores mayores a {percentil}.")

    if estadistico > percentil:
      print("""El estadístico observado cae en la región de rechazo y por lo tanto
rechazamos la hipótesis nula.""")
      hipotesis_nula = False

    else:
      print("""La evidencia observada no alcanza para rechazar la hipótesis nula.""")
      hipotesis_nula = True

    return hipotesis_nula