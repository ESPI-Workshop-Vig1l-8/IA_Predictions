"""
Entraînement de l'Isolation Forest.

    python entrainement.py                      # données recupérées du backend : donnees/normal.csv
    
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from generer_données import colonnes, periode, generer_serie
from caracteristiques import confirmer, fenetres, taille_fenetre

# Init des fichiers et paramètres
dossier = Path(__file__).parent
fichier_donnees = dossier / "donnees" / "normal.csv"
# FICHIER_MODELE : où écrire le modèle (volume Docker /models dans la stack infra)
fichier_modele = Path(os.environ.get("FICHIER_MODELE", dossier / "modele_anomalies.joblib"))
contamination = 0.005
k_confirmation = 5
fenetres_minimum = 50


def fenetres_par_segment(donnees, pas=10):
    """Fenêtres calculées segment par segment : une fenêtre ne chevauche jamais une
    coupure (message perdu, redémarrage, mesure rejetée). Sans colonne "segment"
    (données simulées), toute la série est continue."""
    if "segment" not in donnees:
        return fenetres(donnees[colonnes].to_numpy(), pas=pas)[0]
    blocs = [fenetres(seg[colonnes].to_numpy(), pas=pas)[0]
             for _, seg in donnees.groupby("segment") if len(seg) >= taille_fenetre]
    return np.vstack(blocs) if blocs else np.empty((0, 10))


def entrainer(donnees=None):
    if donnees is None:
        if not fichier_donnees.exists():
            raise SystemExit(f"Fichier introuvable : {fichier_donnees}\n"
                             "Lancez d'abord generer_données.py pour le creer.")
        donnees = pd.read_csv(fichier_donnees)

    x = fenetres_par_segment(donnees, pas=10)
    print(f"{len(donnees)} mesures -> {len(x)} fenetres d'entrainement, {x.shape[1]} caracteristiques")
    if len(x) < fenetres_minimum:
        raise SystemExit(f"Pas assez de données : {len(x)} fenêtres (minimum {fenetres_minimum}). "
                         "Laissez tourner le capteur plus longtemps.")

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


def evaluer_reel(modele, donnees, session, url, appareil, annotations):
    """Évaluation sur les vraies données : fausses alertes sur les données normales,
    et détection des périodes de test annotées."""
    import exporter_couchdb as ex

    print("\n--- Evaluation sur les vraies mesures ---")
    x = fenetres_par_segment(donnees, pas=1)
    if len(x):
        print(f"normal : fenetres jugees anormales {np.mean(modele.predict(x) == -1) * 100:.2f} % "
              f"(alertes confirmees : {confirmer(modele.predict(x) == -1, k_confirmation).mean() * 100:.2f} %)")

    if not annotations:
        print("Aucune periode annotee : faites un test (briquet, souffle chaud) et annotez-le "
              "depuis le dashboard pour mesurer la detection.")
        return
    marge = taille_fenetre * periode * 1000  # contexte avant le test pour remplir la fenêtre
    for debut, fin in annotations:
        docs = ex.lire_mesures(session, url, appareil, debut - marge, fin)
        periode_test, _ = ex.convertir(docs, [], garder_annotations=True, taille_min=taille_fenetre)
        x = fenetres_par_segment(periode_test, pas=1)
        quand = datetime.fromtimestamp(debut / 1000).strftime("%d/%m %H:%M")
        if not len(x):
            print(f"test du {quand} : pas assez de mesures continues")
            continue
        detecte = confirmer(modele.predict(x) == -1, k_confirmation).any()
        print(f"test du {quand} ({(fin - debut) // 1000} s) : {'DETECTE' if detecte else 'non detecte'}")


def main():
    maintenant = datetime.now()
    parser = argparse.ArgumentParser(description="Entraînement de l'Isolation Forest")
    parser.add_argument("--source", choices=["csv", "couchdb"], default="csv",
                        help="csv : donnees/normal.csv (simulé ou exporté) ; couchdb : vraies mesures du backend")
    parser.add_argument("--appareil", default="VIG1L-8-NODE04")
    parser.add_argument("--debut", default=(maintenant - timedelta(days=1)).isoformat(timespec="minutes"),
                        help="date ISO, heure locale (défaut : il y a 24 h)")
    parser.add_argument("--fin", default=maintenant.isoformat(timespec="seconds"))
    args = parser.parse_args()

    if args.source == "csv":
        modele = entrainer()
        if "segment" in pd.read_csv(fichier_donnees, nrows=1).columns:
            print("\nDonnees reelles (export CouchDB) : l'evaluation sur scenarios simules ne s'applique pas, "
                  "utilisez --source couchdb pour evaluer sur les periodes annotees.")
        else:
            evaluer(modele)
        return

    import exporter_couchdb as ex

    session, url = ex.connexion()
    docs = ex.lire_mesures(session, url, args.appareil, ex.date_ms(args.debut), ex.date_ms(args.fin))
    annotations = ex.lire_annotations(session, url, args.appareil)
    donnees, rejets = ex.convertir(docs, annotations)
    print(f"CouchDB : {len(docs)} mesures lues, {len(donnees)} gardees "
          f"en {donnees['segment'].nunique() if len(donnees) else 0} segment(s)")
    for raison, nombre in rejets.items():
        if nombre:
            print(f"  - {nombre} rejetees : {raison}")
    if len(donnees):
        fichier_donnees.parent.mkdir(exist_ok=True)
        donnees.round({c: 2 for c in colonnes}).to_csv(fichier_donnees, index=False)
        print(f"Donnees d'entrainement sauvegardees : {fichier_donnees}")
    modele = entrainer(donnees)
    evaluer_reel(modele, donnees, session, url, args.appareil, annotations)


if __name__ == "__main__":
    main()
