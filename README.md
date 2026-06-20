# Predicción de Goles: Alemania vs Costa de Marfil

## Descripción del Proyecto
Este proyecto utiliza ciencia de datos para predecir el marcador del partido entre Alemania y Costa de Marfil utilizando modelos estadísticos avanzados:

- **Limpieza de datos**: Filtrado desde 2018
- **Modelo Dixon Hub**: Estimación de fuerza ofensiva y defensiva
- **Métodos de predicción**:
  - MCMC (Markov Chain Monte Carlo)
  - Métodos Bayesianos
  - XGBoost
- **Distribución de Poisson**: Modelado de goles esperados
- **Matriz de probabilidades**: Cálculo de probabilidades para cada posible marcador
- **Validación**: Medición de accuracy

## Archivos
- `data_processing.py`: Importación y limpieza de datos
- `dixon_hub_model.py`: Implementación del modelo Dixon Hub
- `prediction_models.py`: Modelos MCMC, Bayesianos y XGBoost
- `poisson_analysis.py`: Análisis con distribución de Poisson
- `main_analysis.py`: Orquestación del análisis completo
- `requirements.txt`: Dependencias del proyecto

## Ejecución
```bash
pip install -r requirements.txt
python main_analysis.py
```
