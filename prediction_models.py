import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
import pymc3 as pm
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

class BayesianPredictionModel:
    """
    Modelo Bayesiano usando MCMC para predecir goles
    """
    
    def __init__(self, df, dixon_model):
        self.df = df
        self.dixon_model = dixon_model
        self.teams = sorted(list(dixon_model.teams))
        
    def fit(self, draws=2000, tune=1000):
        """
        Ajusta el modelo Bayesiano usando MCMC
        """
        print("\nAjustando modelo Bayesiano con MCMC...")
        
        # Preparar datos
        home_teams = self.df['home_team'].values
        away_teams = self.df['away_team'].values
        home_goals = self.df['home_score'].values
        away_goals = self.df['away_score'].values
        
        team_idx = {team: i for i, team in enumerate(self.teams)}
        home_team_idx = np.array([team_idx[t] for t in home_teams])
        away_team_idx = np.array([team_idx[t] for t in away_teams])
        
        with pm.Model() as model:
            # Priors para fuerza ofensiva y defensiva
            attack = pm.Exponential('attack', 1, shape=len(self.teams))
            defense = pm.Exponential('defense', 1, shape=len(self.teams))
            home_adv = pm.Normal('home_advantage', 0, sigma=0.3)
            
            # Goles esperados
            mu_home = attack[home_team_idx] / defense[away_team_idx] * pm.math.exp(home_adv)
            mu_away = attack[away_team_idx] / defense[home_team_idx]
            
            # Verosimilitud (Poisson)
            pm.Poisson('home_goals', mu=mu_home, observed=home_goals)
            pm.Poisson('away_goals', mu=mu_away, observed=away_goals)
            
            # Muestreo
            self.trace = pm.sample(draws=draws, tune=tune, return_inferencedata=True, 
                                   progressbar=True, cores=4)
        
        print("✓ Modelo Bayesiano ajustado correctamente")
        return self
    
    def predict(self, home_team, away_team):
        """
        Predice goles esperados usando el modelo Bayesiano
        """
        team_idx = {team: i for i, team in enumerate(self.teams)}
        
        # Obtener muestras posteriores
        attack_samples = self.trace.posterior['attack'].values.reshape(-1, len(self.teams))
        defense_samples = self.trace.posterior['defense'].values.reshape(-1, len(self.teams))
        home_adv_samples = self.trace.posterior['home_advantage'].values.flatten()
        
        # Calcular goles esperados para cada muestra
        home_idx = team_idx[home_team]
        away_idx = team_idx[away_team]
        
        expected_home = attack_samples[:, home_idx] / defense_samples[:, away_idx] * np.exp(home_adv_samples)
        expected_away = attack_samples[:, away_idx] / defense_samples[:, home_idx]
        
        return np.mean(expected_home), np.mean(expected_away)


class XGBoostPredictionModel:
    """
    Modelo XGBoost para predecir goles
    """
    
    def __init__(self, df, dixon_model):
        self.df = df
        self.dixon_model = dixon_model
        self.teams = sorted(list(dixon_model.teams))
        self.team_encoder = LabelEncoder().fit(self.teams)
        
        # Modelos separados para goles locales y visitantes
        self.model_home = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
        self.model_away = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
        
    def prepare_features(self, df):
        """
        Prepara características para XGBoost
        """
        X = pd.DataFrame()
        X['home_team_encoded'] = self.team_encoder.transform(df['home_team'])
        X['away_team_encoded'] = self.team_encoder.transform(df['away_team'])
        
        # Características del modelo Dixon
        X['home_attack'] = df['home_team'].map(self.dixon_model.attack)
        X['home_defense'] = df['home_team'].map(self.dixon_model.defense)
        X['away_attack'] = df['away_team'].map(self.dixon_model.attack)
        X['away_defense'] = df['away_team'].map(self.dixon_model.defense)
        
        X['home_advantage'] = self.dixon_model.home_advantage
        
        return X
    
    def fit(self):
        """
        Ajusta los modelos XGBoost
        """
        print("\nAjustando modelos XGBoost...")
        
        X = self.prepare_features(self.df)
        y_home = self.df['home_score']
        y_away = self.df['away_score']
        
        self.model_home.fit(X, y_home, verbose=False)
        self.model_away.fit(X, y_away, verbose=False)
        
        print("✓ Modelos XGBoost ajustados correctamente")
        return self
    
    def predict(self, home_team, away_team):
        """
        Predice goles esperados usando XGBoost
        """
        X = pd.DataFrame({
            'home_team_encoded': [self.team_encoder.transform([home_team])[0]],
            'away_team_encoded': [self.team_encoder.transform([away_team])[0]],
            'home_attack': [self.dixon_model.attack[home_team]],
            'home_defense': [self.dixon_model.defense[home_team]],
            'away_attack': [self.dixon_model.attack[away_team]],
            'away_defense': [self.dixon_model.defense[away_team]],
            'home_advantage': [self.dixon_model.home_advantage]
        })
        
        expected_home = self.model_home.predict(X)[0]
        expected_away = self.model_away.predict(X)[0]
        
        return max(0, expected_home), max(0, expected_away)
