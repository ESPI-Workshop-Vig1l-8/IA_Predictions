import argparse
import time
from collections import deque

import joblib
import numpy as np

from generer_données import colonnes, periode, generer_serie
from caracteristiques import taille_fenetre, extraire 

fichier_modele = "modeles_anomalies.joblib"
k_confirmation = 5
seuil_critique_temp = 40 
seuil_critique_gaz = 400 

# exemple de flux fictif

def flux_fictif (scenario, acceleration):
    n = 30*60 // periode
    debut = 5*60 // periode
    serie = generer_serie(n, scenario, debut_anomalie = debut, graine = None)
    for i, ligne in enumerate (serie[colonnes].to_numpy()) :
        time.sleep(periode / acceleration)
        yield i, ligne

def main () : 
    parseur = argparse.ArgumentParser
    parseur.add_argument("--scenario", default="derive", choices=["normal", "derive", "pic_gaz"])
    parseur.add_argument("--acceleration", type=float, default=20)
    args = parseur.parse_args()

    modele = joblib.load(fichier_modele)
    fenetre = deque(maxlen = taille_fenetre)
    consecutives = 0 
    alerte_deja_levee = False

    print (f"Scénario: {args.scenario} | accélération : {args.acceleration}x")

    for i, mesure in flux_fictif(args.scenario, args.acceleration) :
        fenetre.append(mesure)
        temp, hum, gaz = mesure

        if len(fenetre) < taille_fenetre :
            print(f"[{i * periode:5d}s] remplissage de la fenêtre ({len(fenetre)}/{taille_fenetre})", end="\r")
            continue

        x = extraire(np.array(fenetre)).reshape(1, -1)
        score = float (modele.decision_function(x)[0])
        anormal = modele.predict(x)[0] == -1
        consecutives = consecutives + 1 if anormal else 0
        alerte = consecutives >= k_confirmation