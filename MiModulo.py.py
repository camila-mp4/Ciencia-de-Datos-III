import random

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.metrics import auc
from scipy.stats import chi2

class TrainTest():
  """Separa conjunto de datos en conjunto train y test.

  Args:
      data (pd.Dataframe): conjunto de datos a separar
      p (float): porcentaje de datos para conjunto train. 0 < p < 1.
      semilla (int): opcional, permite definir la semilla de selección de datos.
  """
  def __init__(self, data: pd.DataFrame, p:float, semilla:int = 10):
    """Inicializa instancia de la clase TrainTest."""
    random.seed(semilla)

    self.indices = random.sample(range(len(data)), int(len(data) * p))

    self.train = data.iloc[self.indices]
    self.test = data.drop(self.indices)

  def train(self) -> pd.DataFrame:
    """Devuelve conjunto train."""
    return self.train

  def test(self) -> pd.DataFrame:
    """Devuelve conjunto test."""
    return self.test

class Regresion():
  """Realiza cálculos de regresión con librería statmodels.

    Atributos:
        x (np.ndarray): vector de observaciones de la variable predictora.
        y (np.ndarray): vector de observaciones de la variable respuesta.
        X (np.ndarray): matriz de diseño asociada a x."""

  def __init__(self, predictoras, respuesta):
    """Inicializa instancia de clase regresión."""
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
  """Esta clase es una instancia de la clase Regresion y realiza cálculos y gráficos de regresión lineal
  utilizando las librerías numpy, statsmodels y matplotlib.

  Atributos:
  - Modelo (sm.OLS): Modelo de regresión lineal.
  - """

  def __init__(self, predictoras: np.ndarray, respuesta: np.ndarray):
    """ Inicializa una instancia de la clase Regresión Lineal con la librería
    statsmodels.

    Atributos:
    modelo: instancia de la clase statmodels.OLS
    resultado:
    """
    super().__init__(predictoras, respuesta)
    self.modelo = sm.OLS(self.y, self.X)
    self.resultado = self.modelo.fit()

  def coeficientes_correlacion(self):
    """ Devuelve los coeficientes de correlación entre cada variable predictora
    y la variable respuesta.

    Returns:
        np.ndarray: Coeficientes de correlación entre x e y.
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
    '''Realiza un gráfico de dispersión para cada variable predictora.
    En caso de haber una sola, superpone la recta ajustada.'''

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
            np.ndarray: predicciones
    """
    X_new = np.insert(x_new, 0, 1)
    return X_new @ self.betas()

  def ECM(self, x_test: np.ndarray, y_test:np.ndarray) -> float:
    """Devuelve el error cuadrático medio asociado a una muestra testeo.
    Args:
        x_test (np.ndarray): predictoras para testear.
        y_test (np.ndarray): valores de la variable respuesta para testear.

    Returns:
        float"""
    error_cuadratico_medio = np.sum((y_test - self.predecir_valor(x_test))**2) / len(self.y)
    return error_cuadratico_medio

  def intervalos(self, x_new, alfa=0.05, mostrar = True):
    '''Devuelve intervalos de confianza y de predicción para un valor
    de las variables predictoras.
      Argumentos:'''
    X_new = np.insert(x_new, 0, 1)
    prediccion = self.resultado.get_prediction(X_new)

    int_confianza = prediccion.conf_int(alpha=alfa)
    int_prediccion = prediccion.conf_int(obs = True, alpha = alfa)

    if mostrar == True:
      print(f"El intervalo de confianza para mu_(Y| {x_new}) es: {int_confianza[0]}")
      print(f"El intervalo de predicción para X = x_new es: {int_prediccion[0]}")

    return int_confianza[0], int_prediccion[0]

  def testeo(self, x_test, y_test):
    X_test = sm.add_constant(x_test)
    y_pred = X_test @ self.betas
    diferencias = (np.sum(((y_pred - y_test)**2) / len(y_pred))) ** 0.5
    return diferencias

  def analisis_residuos(self):
    residuos = self.resultado.resid
    plt.title("Residuos vs. valores predichos.")
    plt.scatter(self.valores_ajustados(), self.resultado.resid)
    plt.show()

    residuos_normalizados = (np.sort(residuos) - np.mean(residuos)) / np.std(residuos)

    sm.qqplot(residuos_normalizados, line='45')
    plt.title('QQplot normal de los residuos')

class RegresionLogistica(Regresion):
  def __init__(self,predictoras,respuesta):
    super().__init__(predictoras, respuesta)
    self.modelo = sm.Logit(self.y, self.X)
    self.resultado = self.modelo.fit()
    self.probabilidades_estimadas = np.e ^ self.valores_ajustados / (np.e ^ self.valores_ajustados + 1)

  def estimar_probabilidad(self, x_test):
    X_test = sm.add_constant(x_test)
    exp = X_test @ self.betas
    probabilidades = np.e ^ exp / (1 + np.e ^ exp)
    return probabilidades

  def categoriza(self, x_test, p = 0.05):
    categorizadas = 1 * (self.estimar_probabilidad(x_test) >= p)
    return categorizadas

  def matriz_confusion(self, x_test, y_test, p = 0.05, mostrar = True):
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

  def ROC(self, x_test, y_test):
    p = np.linspace(0, 1, 100)

    sensibilidad = np.zeros(len(p))
    especificidad = np.zeros(len(p))

    for i in range(len(p)):
      sensibilidad[i] = self.matriz_confusion(x_test, y_test, p[i], mostrar = False)['sensibilidad']
      especificidad[i] = self.matriz_confusion(x_test, y_test, p[i], mostrar = False)['especificidad']

    curva_ROC = plt.plot(1 - especificidad, sensibilidad)

    AUC = auc(1 - especificidad, sensibilidad)

    if AUC <= 0.6:
      print('Modelo fallido.')
    elif 0.6 < AUC <= 0.7:
      print('Modelo pobre')
    elif 0.7 < AUC <= 0.8:
      print('Modelo regular.')
    elif 0.8 < AUC <= 0.9:
      print('Modelo bueno.')
    else:
      print('Modelo excelente.')

    return curva_ROC, AUC