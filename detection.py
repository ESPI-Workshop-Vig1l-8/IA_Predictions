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



