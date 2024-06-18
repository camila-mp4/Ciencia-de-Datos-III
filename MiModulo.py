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
    bins = np.arange(min(self.datos),max(self.datos), h)
    frecuencias = np.zeros(len(bins))

    for i in range(len(bins)):
      if i+1 == len(bins):
        break
      ind_m = np.where(self.datos >= bins[i])
      ind_M = np.where(self.datos < bins[i+1])
      ind_bins = np.intersect1d(ind_m, ind_M)

      frecuencias[i] = len(ind_bins)

    densidad = frecuencias / (len(self.datos) * h)

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

class ResumenNumerico():
  """Clase que realiza el resumen numérico de una muestra de una variable
  cuantitativa. Calcula media, desvío, cuartiles, mínimo y máximo de la misma.
  Args:
      datos: vector con las observaciones de la muestra."""
  def __init__(self, datos):
    self.datos = datos

  def __str__(self):
    """Retorno de aplicar print() a un objeto de la clase ResumenNumérico."""
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

  def train(self) -> pd.DataFrame:
    """Devuelve conjunto de entrenamiento."""
    return self.train

  def test(self) -> pd.DataFrame:
    """Devuelve conjunto de testeo."""
    return self.test

class Regresion():
  """Realiza cálculos de regresión con librería statmodels.

    Atributos:
        x (np.ndarray): vector de observaciones de la variable predictora.
        y (np.ndarray): vector de observaciones de la variable respuesta.
        X (np.ndarray): matriz de diseño asociada a x."""

  def __init__(self, predictoras, respuesta):
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

class RegresionLineal(Regresion):
  """Esta clase es una instancia de la clase Regresion y realiza cálculos
<<<<<<< HEAD
  y gráficos de regresión lineal utilizando las librerías numpy, statsmodels
  y matplotlib.

  Atributos:
  - Modelo (sm.OLS): Modelo de regresión lineal.
  - """
=======
  y gráficos para modelos de regresión lineal utilizando las librerías numpy,
  statsmodels y matplotlib.
  """
>>>>>>> nueva_rama

  def __init__(self, predictoras: np.ndarray, respuesta: np.ndarray):
    """ Inicializa una instancia de la clase Regresión Lineal con la librería
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

    if self.x.ndim == 1:
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

  def predecir_valor(self, x_new):
    """Predice el valor  de la variable respuesta asociado a nuevos valores
    de las variables predictoras.
      Args:
            x_new: vector de variables predictoras.
      Returns:
            np.ndarray: predicciones.
    """
    X_new = np.insert(x_new, 0, 1)
    return X_new @ self.betas()

  def ECM(self, x_test: np.ndarray, y_test:np.ndarray) -> float:
    """Devuelve el error cuadrático medio asociado a una muestra de testeo.
    Args:
        x_test (np.ndarray): predictoras para testear.
        y_test (np.ndarray): valores de la variable respuesta para testear.

    Returns:
        error_cuadratico_medio (float): estimación del ECM del modelo.
<<<<<<< HEAD
        """
=======
    """
>>>>>>> nueva_rama
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
<<<<<<< HEAD
          """
=======
    """
>>>>>>> nueva_rama
    X_new = np.insert(x_new, 0, 1)
    prediccion = self.resultado.get_prediction(X_new)

    int_confianza = prediccion.conf_int(alpha=alfa)
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
    self.modelo = sm.Logit(self.y, self.X)
    self.resultado = self.modelo.fit()
    self.probabilidades_estimadas = np.e ^ self.valores_ajustados / (np.e ^ self.valores_ajustados + 1)

  def estimar_probabilidad(self, x_test: np.ndarray) -> np.ndarray:
    """Dado un vector de prueba x_test, estima la probabilidad de que la variable
    respuesta tome tal valor con el modelo ajustado.

    Args:
        x_test (np.ndarray): vector de valores independientes de las predictoras.
    """
    X_test = sm.add_constant(x_test)
    exp = X_test @ self.betas
    probabilidades = np.e ^ exp / (1 + np.e ^ exp)
    return probabilidades

  def categoriza(self, x_test: np.ndarray, p: float = 0.05) -> np.ndarray:
    """Dado un umbral p y un vector de prueba x_test, categoriza como 1 aquellas
    probabilidades mayores a p y 0 a las demás.
    Args:
        x_test (np.ndarray): vector de valores independientes de las predictoras.
        p (float): punto de corte para clasificar.
    """
    categorizadas = 1 * (self.estimar_probabilidad(x_test) >= p)
    return categorizadas

  def matriz_confusion(self, x_test: np.ndarray, y_test: np.ndarray, p: float = 0.05, mostrar: bool = True) -> dict:
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
    y_pred = self.categoriza(x_test, p)

    a = verdaderos_positivos = np.sum((y_pred == 1) & (y_test == 1))
    b = falsos_positivos = np.sum((y_pred == 1) & (y_test == 0))
    c = falsos_negativos = np.sum((y_pred == 0) & (y_test == 1))
    d = verdaderos_negativos = np.sum((y_pred == 0) & (y_test == 0))

    tabla =  pd.DataFrame({
    'y_test = 1': [a, c],
    'y_test = 0': [b, d],
    }, index=['y_pred = 1', 'y_pred = 0'])

    error_mala_clasificacion = (b + c) / len(y_pred)
    sensibilidad = a / (a + c)
    especificidad = d / (b + d)

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
    p = np.linspace(0, 1, 100)

    sensibilidad = np.zeros(len(p)) # sensibilidad para cada valor de p
    especificidad = np.zeros(len(p)) # especificidad para cada valor de p

    for i in range(len(p)):
      sensibilidad[i] = self.matriz_confusion(x_test, y_test, p[i], mostrar = False)['sensibilidad']
      especificidad[i] = self.matriz_confusion(x_test, y_test, p[i], mostrar = False)['especificidad']
    
    return sensibilidad, especificidad

  def ROC(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
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
    x, y = self.ROC(x_test, y_test)
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

    AUC = self.AUC_modelo(x_test, y_test)

    if AUC <= 0.6:
      clasificacion = 'Modelo fallido.'
    elif 0.6 < AUC <= 0.7:
      clasificacion = 'Modelo pobre.'
    elif 0.7 < AUC <= 0.8:
      clasificacion = 'Modelo regular.'
    elif 0.8 < AUC <= 0.9:
      clasificacion = 'Modelo bueno.'
    else:
      clasificacion = 'Modelo excelente.'

<<<<<<< HEAD
    return curva_ROC, AUC
=======
    return clasificacion
>>>>>>> nueva_rama
