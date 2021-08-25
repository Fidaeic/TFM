# TFM
Repositorio para el almacenamiento del código y los resultados relativos al Trabajo de Fin de Máster del Máster de Ingeniería de Análisis de Datos, Mejora de Procesos y Toma de Decisiones, elaborado por Fidae El Morer. El título del trabajo es "ELABORACIÓN DE UNA METODOLOGÍA DE TRABAJO PARA EL TRATAMIENTO Y LA PREDICCIÓN DE SERIES TEMPORALES DE CONSUMO DE AGUA POTABLE". La estructura del repositorio es como sigue:

* **Código**:
  * main.py: Script principal, en el que se ejecutan todas las posibles combinaciones de métodos desarrollados en waTS.py, y que sirve para decidir la estrategia óptima de imputación, reconstrucción y predicción.
  * waTS.py: Se trata del fichero principal, en el que se encuentran implementados todos los objetos, así como los métodos y atributos asociados para el desarrollo de los cálculos.
  * ga.py: En este fichero se encuentra la implementación del algoritmo genético desarrollado para la optimización de los hiperparámetros de los algoritmos de predicción.
  * ga_example: Ejemplo de la optimización de un algoritmo mediante el uso del script desarrollado en ga.py.
  * metrics.py: Fichero en el que se recogen algunas métricas de validación utilizadas a lo largo del trabajo.
  * results.py: Script en el que se analizan los resultados mediante gráficos y pruebas de hipótesis.
* **Data**: Carpeta en la que se almacenan los datos de la serie temporal original, previamente a su manipulación
* **Results**: Carpeta en la que se almacenan los datos reales que se intentan predecir, los resultados de todas las combinaciones y los tiempos de ejecución. También se incluye un fichero en el que se encuentran los resultados del mejor modelo obtenido, junto con un intervalo de confianza del 95%.
