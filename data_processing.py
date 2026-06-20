import pandas as pd
import numpy as np
from datetime import datetime
import requests
from io import StringIO

def download_data():
    """
    Descarga el archivo results.csv del repositorio martj42/international_results
    """
    print("Descargando datos de partidos internacionales...")
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        print(f"✓ Datos descargados: {len(df)} registros")
        return df
    except Exception as e:
        print(f"✗ Error descargando datos: {e}")
        return None

def clean_data(df):
    """
    Limpia y prepara los datos para el análisis
    """
    print("\nLimpiando datos...")
    
    # Convertir fecha a datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filtrar desde 2018
    df_filtered = df[df['date'].dt.year >= 2018].copy()
    print(f"✓ Datos filtrados desde 2018: {len(df_filtered)} partidos")
    
    # Eliminar registros con valores faltantes en columnas clave
    df_filtered = df_filtered.dropna(subset=['home_team', 'away_team', 'home_score', 'away_score'])
    print(f"✓ Registros válidos: {len(df_filtered)}")
    
    # Asegurar tipos de datos correctos
    df_filtered['home_score'] = df_filtered['home_score'].astype(int)
    df_filtered['away_score'] = df_filtered['away_score'].astype(int)
    
    return df_filtered

def get_team_stats(df, team_name):
    """
    Calcula estadísticas de un equipo
    """
    home_matches = df[df['home_team'] == team_name].copy()
    away_matches = df[df['away_team'] == team_name].copy()
    
    stats = {
        'team': team_name,
        'home_matches': len(home_matches),
        'away_matches': len(away_matches),
        'home_goals': home_matches['home_score'].sum(),
        'home_goals_against': home_matches['away_score'].sum(),
        'away_goals': away_matches['away_score'].sum(),
        'away_goals_against': away_matches['home_score'].sum(),
    }
    
    stats['home_goals_per_match'] = stats['home_goals'] / max(stats['home_matches'], 1)
    stats['away_goals_per_match'] = stats['away_goals'] / max(stats['away_matches'], 1)
    stats['home_goals_against_per_match'] = stats['home_goals_against'] / max(stats['home_matches'], 1)
    stats['away_goals_against_per_match'] = stats['away_goals_against'] / max(stats['away_matches'], 1)
    
    return stats

if __name__ == "__main__":
    df = download_data()
    if df is not None:
        df_clean = clean_data(df)
        
        # Ejemplo: estadísticas de Alemania
        germany_stats = get_team_stats(df_clean, 'Germany')
        print("\nEstadísticas de Alemania:")
        for key, value in germany_stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
