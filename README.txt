# Biblioteca de Ciencia de Datos.

Módulo de Python para cálculos estadísticos.

## Descripción.

Esta librería otorga herramientas para la realización de ciertos cálculos comunes en la Ciencia de Datos, tales como regresión lineal, regresión logística, test chi2 para variables cualitativas, etc.

## Características.

- Clase ResumenNumérico: dado un conjunto de datos numéricos devuelve estadísticos observados en el mismo, tales como la media, el desvío estándar, entre otros.
- Clase ResumenGráfico: dado un conjunto de observaciones de una variable cuantitativa halla su histograma y estima su densidad por kernels.
- Clase TrainTest: dado un conjunto de datos, lo separa en conjuntos para entrenar y testear un modelo, permitiendo especificar el porcentaje de datos que formarán cada conjunto y la semilla utilizada.
- Clase Regresión: clase básica para modelos de regresión.
- Clase RegresiónLineal: clase que hereda de Regresión y realiza cálculos de regresión lineal simple y múltiple con la librería statmodels.
- Clase RegresiónLogística: clase que hereda de Regresión y realiza cálculo de regresión logística con librería statmodels.
- Clase test_chi2: clase que, dado un vector de probabilidades teórico y muestra de observaciones de una variable cualitativa, realiza test chi2 para determinar si la muestra sigue la distribución de probabilidad propuesta.

## Instalación

1. Clonar el siguiente repositorio:
https://github.com/camila-mp4/Ciencia-de-Datos-III

2. Ejecutar el archivo 
MiModulo.py