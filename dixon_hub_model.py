import pandas as pd
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict

class DixonHubModel:
    """
    Modelo Dixon Hub para estimar la fuerza ofensiva y defensiva de cada selección.
    Basado en el modelo propuesto por Dixon & Coles (1997)
    """
    
    def __init__(self, df):
        self.df = df
        self.teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        self.team_list = sorted(list(self.teams))
        self.team_idx = {team: idx for idx, team in enumerate(self.team_list)}
        
        # Parámetros a estimar
        self.attack = {team: 1.0 for team in self.teams}
        self.defense = {team: 1.0 for team in self.teams}
        self.home_advantage = 0.0
        
    def poisson_loglik(self, y, mu):
        """Log-verosimilitud de Poisson"""
        return np.sum(y * np.log(mu) - mu - np.log(np.math.factorial(int(y))))
    
    def estimate_parameters(self):
        """
        Estima los parámetros del modelo usando máxima verosimilitud
        """
        print("Estimando parámetros del modelo Dixon Hub...")
        
        # Inicializar parámetros
        n_teams = len(self.team_list)
        x0 = np.concatenate([
            np.ones(n_teams),  # attack
            np.ones(n_teams),  # defense
            [0.0]  # home_advantage
        ])
        
        # Función objetivo (negativa log-verosimilitud)
        def objective(params):
            attack = params[:n_teams]
            defense = params[n_teams:2*n_teams]
            home_adv = params[2*n_teams]
            
            ll = 0
            for _, row in self.df.iterrows():
                home_idx = self.team_idx[row['home_team']]
                away_idx = self.team_idx[row['away_team']]
                
                # Goles esperados
                mu_home = np.exp(attack[home_idx] - defense[away_idx] + home_adv)
                mu_away = np.exp(attack[away_idx] - defense[home_idx])
                
                # Log-verosimilitud
                ll += (row['home_score'] * np.log(mu_home) - mu_home)
                ll += (row['away_score'] * np.log(mu_away) - mu_away)
            
            return -ll
        
        # Optimizar
        result = minimize(objective, x0, method='Nelder-Mead')
        
        # Asignar parámetros estimados
        opt_params = result.x
        for i, team in enumerate(self.team_list):
            self.attack[team] = np.exp(opt_params[i])
            self.defense[team] = np.exp(opt_params[n_teams + i])
        self.home_advantage = opt_params[2*n_teams]
        
        print("✓ Parámetros estimados correctamente")
        return self
    
    def predict(self, home_team, away_team):
        """
        Predice los goles esperados para un partido
        """
        expected_home_goals = self.attack[home_team] / self.defense[away_team] * np.exp(self.home_advantage)
        expected_away_goals = self.attack[away_team] / self.defense[home_team]
        
        return expected_home_goals, expected_away_goals
    
    def print_rankings(self):
        """
        Imprime los equipos ordenados por fuerza ofensiva
        """
        print("\n=== Rankings por Fuerza Ofensiva ===")
        sorted_teams = sorted(self.attack.items(), key=lambda x: x[1], reverse=True)
        for i, (team, attack_power) in enumerate(sorted_teams[:10], 1):
            defense_power = self.defense[team]
            print(f"{i}. {team:20} - Ataque: {attack_power:.4f}, Defensa: {defense_power:.4f}")
