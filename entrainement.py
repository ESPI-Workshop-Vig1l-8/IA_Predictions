from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from generer_données import colonnes, periode, generer_serie
from caracteristiques import confirmer, fenetres

# Init des fichiers et paramètres
dossier = Path(__file__).parent
fichier_donnees = dossier / "donnees" / "normal.csv"
fichier_modele = dossier / "modele_anomalies.joblib"   
contamination = 0.005
k_confirmation = 5


def entrainer():
    if not fichier_donnees.exists():
        raise SystemExit(f"Fichier introuvable : {fichier_donnees}\n"
                         "Lancez d'abord generer_données.py pour le creer.")

    donnees = pd.read_csv(fichier_donnees)
    x, _ = fenetres(donnees[colonnes].to_numpy(), pas=10)
    print(f"{len(donnees)} mesures -> {len(x)} fenetres d'entrainement, {x.shape[1]} caracteristiques")

    modele = make_pipeline(
        StandardScaler(),   # Met les caractéristiques à la même échelle
        IsolationForest(n_estimators=200, contamination=contamination, random_state=42),
    )
    modele.fit(x)
    joblib.dump(modele, fichier_modele)
    print(f"Modele sauvegarde : {fichier_modele}")
    return modele


# Tester le modèle sur des séries de données neuves
def evaluer(modele):
    print("\n--- Evaluation sur donnees simulées neuves ---")
    n = 20 * 60 // periode
    debut = n // 3

    for scenario in ["normal", "derive", "pic_gaz"]:
        taux, delais = [], []

        for graine in range(1, 11):
            serie = generer_serie(n, scenario, debut_anomalie=debut, graine=graine)
            X, fins = fenetres(serie[colonnes].to_numpy())
            brut = modele.predict(X) == -1
            alerte = confirmer(brut, k_confirmation)

            if scenario == "normal":
                taux.append(alerte.mean())
            else:
                apres = fins >= debut
                idx = np.where(alerte & apres)[0]
                if len(idx):
                    delais.append((fins[idx[0]] - debut) * periode / 60)
                taux.append(len(idx) > 0)

        # Affichage une fois par scénario, donc en dehors de la boucle sur les graines
        if scenario == "normal":
            print(f"normal   : fausses alertes sur {np.mean(taux) * 100:.2f} % des fenetres")
        else:
            detectes = int(np.sum(taux))
            moy = f"{np.mean(delais):.1f} min" if delais else "n/a"
            print(f"{scenario:8} : detecte {detectes}/10 fois, delai moyen après le debut : {moy}")


if __name__ == "__main__":
    evaluer(entrainer())