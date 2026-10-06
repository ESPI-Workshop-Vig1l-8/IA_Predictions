import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from generer_données import colonnes, periode, generer_serie
from caracteristiques import taille_fenetre, confirmer, fenetres

fichier_donnes = "donnees/normal.csv"
fichier_modele = "modeles_anomalies.joblib"
contamination = 0.005
k_confirmation = 5

def entrainer () : 
    donnees = pd.read_csv(fichier_donnes)
    x, _ = fenetres(donnees[colonnes].to_numpy(), pas = 10)
    print(f"{len(donnees)} mesures -> {len(x)} fenêtres d'entraînement, {x.shape[1]} caractéristiques")

    modele = make_pipeline(
        StandardScaler(),   # met les caractéristiques à la même échelle
        IsolationForest(n_estimators=200, contamination=contamination, random_state=42),
    )
    modele.fit(x)
    joblib.dump(modele, fichier_modele)
    print(f"Modèle sauvegardé : {fichier_modele}")
    return modele





