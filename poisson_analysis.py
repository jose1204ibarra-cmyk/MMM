import numpy as np
from scipy.stats import poisson
import pandas as pd

class PoissonScoreMatrix:
    """
    Genera matriz de probabilidades de marcadores usando distribución de Poisson
    """
    
    def __init__(self, lambda_home, lambda_away, max_goals=10):
        """
        lambda_home: goles esperados del equipo local
        lambda_away: goles esperados del equipo visitante
        max_goals: máximo número de goles a considerar
        """
        self.lambda_home = lambda_home
        self.lambda_away = lambda_away
        self.max_goals = max_goals
        self.matrix = self._compute_matrix()
        
    def _compute_matrix(self):
        """
        Calcula la matriz de probabilidades
        """
        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
        
        for h_goals in range(self.max_goals + 1):
            for a_goals in range(self.max_goals + 1):
                # Probabilidad conjunta usando distribución de Poisson
                prob_home = poisson.pmf(h_goals, self.lambda_home)
                prob_away = poisson.pmf(a_goals, self.lambda_away)
                matrix[h_goals, a_goals] = prob_home * prob_away
        
        # Normalizar
        matrix = matrix / np.sum(matrix)
        return matrix
    
    def get_match_outcome_probabilities(self):
        """
        Calcula probabilidades de victoria local, empate y victoria visitante
        """
        home_win = 0
        draw = 0
        away_win = 0
        
        for h_goals in range(self.max_goals + 1):
            for a_goals in range(self.max_goals + 1):
                prob = self.matrix[h_goals, a_goals]
                
                if h_goals > a_goals:
                    home_win += prob
                elif h_goals == a_goals:
                    draw += prob
                else:
                    away_win += prob
        
        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win
        }
    
    def get_top_scores(self, top_n=10):
        """
        Retorna los marcadores más probables
        """
        scores = []
        for h_goals in range(self.max_goals + 1):
            for a_goals in range(self.max_goals + 1):
                scores.append({
                    'home_goals': h_goals,
                    'away_goals': a_goals,
                    'probability': self.matrix[h_goals, a_goals]
                })
        
        df_scores = pd.DataFrame(scores).sort_values('probability', ascending=False)
        return df_scores.head(top_n)
    
    def print_matrix(self):
        """
        Imprime la matriz de forma legible
        """
        print("\n=== Matriz de Probabilidades de Marcadores ===")
        print("(Filas: goles locales, Columnas: goles visitantes)\n")
        
        # Encabezado de columnas
        header = "      "
        for col in range(self.max_goals + 1):
            header += f"{col:7d} "
        print(header)
        
        # Filas
        for row in range(self.max_goals + 1):
            line = f"  {row}:  "
            for col in range(self.max_goals + 1):
                line += f"{self.matrix[row, col]:7.4f} "
            print(line)
    
    def print_summary(self, home_team="Local", away_team="Visitante"):
        """
        Imprime un resumen de probabilidades
        """
        outcomes = self.get_match_outcome_probabilities()
        top_scores = self.get_top_scores(5)
        
        print(f"\n=== Predicción de Probabilidades: {home_team} vs {away_team} ===")
        print(f"\nGoles Esperados:")
        print(f"  {home_team}: {self.lambda_home:.2f}")
        print(f"  {away_team}: {self.lambda_away:.2f}")
        
        print(f"\nProbabilidades de Resultado:")
        print(f"  Victoria {home_team}: {outcomes['home_win']*100:.2f}%")
        print(f"  Empate: {outcomes['draw']*100:.2f}%")
        print(f"  Victoria {away_team}: {outcomes['away_win']*100:.2f}%")
        
        print(f"\nMarcadores Más Probables:")
        for idx, row in top_scores.iterrows():
            print(f"  {row['home_goals']}-{row['away_goals']}: {row['probability']*100:.2f}%")
