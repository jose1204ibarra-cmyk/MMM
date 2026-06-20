import numpy as np
import pandas as pd
from data_processing import download_data, clean_data
from dixon_hub_model import DixonHubModel
from prediction_models import BayesianPredictionModel, XGBoostPredictionModel
from poisson_analysis import PoissonScoreMatrix
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    print("="*70)
    print("PREDICCIÓN DE MARCADOR: ALEMANIA VS COSTA DE MARFIL")
    print("="*70)
    
    # Paso 1: Descargar y limpiar datos
    print("\n[PASO 1] Descargando y limpiando datos...")
    df = download_data()
    if df is None:
        return
    
    df_clean = clean_data(df)
    
    # Paso 2: Modelo Dixon Hub
    print("\n[PASO 2] Aplicando modelo Dixon Hub...")
    dixon_model = DixonHubModel(df_clean)
    dixon_model.estimate_parameters()
    dixon_model.print_rankings()
    
    # Predicción con Dixon
    print("\n[PASO 3] Predicciones con Modelo Dixon...")
    home_team = "Germany"
    away_team = "Ivory Coast"
    
    try:
        expected_home_dixon, expected_away_dixon = dixon_model.predict(home_team, away_team)
        print(f"✓ {home_team} vs {away_team}")
        print(f"  Goles esperados - Local: {expected_home_dixon:.2f}, Visitante: {expected_away_dixon:.2f}")
    except Exception as e:
        print(f"✗ Error en predicción Dixon: {e}")
        expected_home_dixon, expected_away_dixon = 1.5, 1.0
    
    # Paso 3: Modelo Bayesiano MCMC
    print("\n[PASO 4] Ajustando modelo Bayesiano (MCMC)...")
    try:
        bayes_model = BayesianPredictionModel(df_clean, dixon_model)
        bayes_model.fit(draws=1000, tune=500)
        expected_home_bayes, expected_away_bayes = bayes_model.predict(home_team, away_team)
        print(f"✓ Predicción Bayesiana:")
        print(f"  Goles esperados - Local: {expected_home_bayes:.2f}, Visitante: {expected_away_bayes:.2f}")
    except Exception as e:
        print(f"✗ Error en modelo Bayesiano: {e}")
        expected_home_bayes, expected_away_bayes = expected_home_dixon, expected_away_dixon
    
    # Paso 4: Modelo XGBoost
    print("\n[PASO 5] Ajustando modelo XGBoost...")
    try:
        xgb_model = XGBoostPredictionModel(df_clean, dixon_model)
        xgb_model.fit()
        expected_home_xgb, expected_away_xgb = xgb_model.predict(home_team, away_team)
        print(f"✓ Predicción XGBoost:")
        print(f"  Goles esperados - Local: {expected_home_xgb:.2f}, Visitante: {expected_away_xgb:.2f}")
    except Exception as e:
        print(f"✗ Error en modelo XGBoost: {e}")
        expected_home_xgb, expected_away_xgb = expected_home_dixon, expected_away_dixon
    
    # Paso 5: Ensamble de predicciones
    print("\n[PASO 6] Ensamble de predicciones...")
    expected_home_ensemble = np.mean([expected_home_dixon, expected_home_bayes, expected_home_xgb])
    expected_away_ensemble = np.mean([expected_away_dixon, expected_away_bayes, expected_away_xgb])
    
    print(f"✓ Predicción Ensamble (promedio de 3 modelos):")
    print(f"  Goles esperados - {home_team}: {expected_home_ensemble:.2f}")
    print(f"  Goles esperados - {away_team}: {expected_away_ensemble:.2f}")
    
    # Paso 6: Análisis con distribución de Poisson
    print("\n[PASO 7] Generando matriz de probabilidades (Poisson)...")
    poisson_matrix = PoissonScoreMatrix(expected_home_ensemble, expected_away_ensemble, max_goals=8)
    poisson_matrix.print_summary(home_team, away_team)
    poisson_matrix.print_matrix()
    
    # Paso 7: Cálculo de Accuracy
    print("\n[PASO 8] Evaluación de modelos...")
    
    # Usar datos históricos para validación
    test_matches = df_clean[df_clean['home_team'].isin([home_team, away_team]) | 
                           df_clean['away_team'].isin([home_team, away_team])]
    
    if len(test_matches) > 0:
        print(f"✓ Partidos históricos encontrados: {len(test_matches)}")
        
        # Predicciones para validación
        predictions_dixon = []
        predictions_bayes = []
        predictions_xgb = []
        actual_scores = []
        
        for _, row in test_matches.head(10).iterrows():
            h_team = row['home_team']
            a_team = row['away_team']
            
            try:
                h_dixon, a_dixon = dixon_model.predict(h_team, a_team)
                predictions_dixon.append((h_dixon, a_dixon))
            except:
                predictions_dixon.append((1.5, 1.0))
            
            predictions_bayes.append((expected_home_bayes, expected_away_bayes))
            predictions_xgb.append((expected_home_xgb, expected_away_xgb))
            actual_scores.append((row['home_score'], row['away_score']))
        
        # Calcular MAE para validación
        if predictions_dixon:
            h_pred = [p[0] for p in predictions_dixon]
            a_pred = [p[1] for p in predictions_dixon]
            h_actual = [s[0] for s in actual_scores]
            a_actual = [s[1] for s in actual_scores]
            
            mae_dixon = mean_absolute_error(h_actual + a_actual, h_pred + a_pred)
            print(f"\n  MAE Modelo Dixon: {mae_dixon:.4f}")
    
    # Reporte final
    print("\n" + "="*70)
    print("RESUMEN FINAL DE PREDICCIÓN")
    print("="*70)
    print(f"\nPartido: {home_team} vs {away_team}")
    print(f"\nPredicciones por modelo:")
    print(f"  Dixon Hub:  {expected_home_dixon:.2f} - {expected_away_dixon:.2f}")
    print(f"  Bayesiano:  {expected_home_bayes:.2f} - {expected_away_bayes:.2f}")
    print(f"  XGBoost:    {expected_home_xgb:.2f} - {expected_away_xgb:.2f}")
    print(f"  Ensamble:   {expected_home_ensemble:.2f} - {expected_away_ensemble:.2f}")
    
    outcomes = poisson_matrix.get_match_outcome_probabilities()
    print(f"\nProbabilidades de resultado:")
    print(f"  Victoria {home_team}: {outcomes['home_win']*100:.2f}%")
    print(f"  Empate: {outcomes['draw']*100:.2f}%")
    print(f"  Victoria {away_team}: {outcomes['away_win']*100:.2f}%")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
