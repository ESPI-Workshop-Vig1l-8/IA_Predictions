import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from generer_données import colonnes, periode, generer_serie
from caracteristiques import taille_fenetre, confirmer, fenetres

# init des fichies et paramètres
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

# tester le modele sur une série de données
def evaluer (modele) : 
    n = 20 * 60 // periode
    debut = n // 3

    for scenario in ["normal", "derive", "pic_gaz"] : 
       taux, delais =[], []
       for graine in range (1, 11) :
           serie = generer_serie(n, scenario, debut_anomalie = debut, graine = graine)
           X, fins = fenetres(serie[colonnes].to_numpy())
           brut = modele.predict(X) == -1
           alerte = confirmer(brut, k_confirmation)

           if scenario == "normal" :
               taux.append(alerte.mean())
           else :
                apres = fins >= debut
                idx = np.where(alerte & apres)[0]
                if len(idx):
                    delais.append((fins[idx[0]] - debut) * periode / 60)  
                taux.append(len(idx) > 0)

           if scenario == "normal":
            print(f"normal   : fausses alertes sur {np.mean(taux) * 100:.2f} % des fenêtres")
           else:
            detectes = int(np.sum(taux))
            moy = f"{np.mean(delais):.1f} min" if delais else "n/a"
            print(f"{scenario:8} : détecté {detectes}/10 fois, délai moyen après le début : {moy}")


if __name__ == "__main__":
    evaluer(entrainer())