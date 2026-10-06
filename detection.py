"""
Détection d'anomalies en continu.

    python detection.py --source mqtt                 # vraies mesures, alertes envoyées au backend
    python detection.py --scenario derive             # simulation (pas d'envoi)
    python detection.py --scenario derive --envoyer   # simulation, alertes envoyées au backend

Deux niveaux d'alerte, une fois chacun par épisode :
  - "warning"   : k_avertissement fenêtres anormales consécutives (début d'un motif anormal)
  - "confirmed" : k_confirmation fenêtres consécutives (le motif se poursuit : la LED du nœud clignote)
L'épisode se termine après fin_episode fenêtres normales consécutives.
"""

import argparse
import os
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np

from generer_données import colonnes, periode, generer_serie
from caracteristiques import taille_fenetre, extraire

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Dossier dans lequel se trouve detection.py
DOSSIER_PROJET = Path(__file__).resolve().parent

# Chemin complet vers le modèle : FICHIER_MODELE s'il existe (modèle réentraîné,
# volume Docker), sinon le modèle livré avec le dépôt
modele_livre = DOSSIER_PROJET / "modele_anomalies.joblib"
modele_entraine = Path(os.environ.get("FICHIER_MODELE", modele_livre))
fichier_modele = modele_entraine if modele_entraine.exists() else modele_livre

k_avertissement = 2
k_confirmation = 5
fin_episode = 15  # 15 fenêtres normales (30 s) avant de pouvoir relancer une alerte

seuil_critique_temp = 40
seuil_critique_gaz = 400


# flux de données fictives pour tester le modèle


def flux_fictif(scenario, acceleration):
    """
    Génère un flux de mesures fictives.

    scenario :
        - normal
        - derive
        - pic_gaz

    acceleration :
        vitesse de simulation.
        Exemple : 20 = simulation 20 fois plus rapide.
    """

    if acceleration <= 0:
        raise ValueError("L'accélération doit être supérieure à 0.")

    n = 30 * 60 // periode
    debut = 5 * 60 // periode

    serie = generer_serie(
        n,
        scenario,
        debut_anomalie=debut,
        graine=None
    )

    for ligne in serie[colonnes].to_numpy():
        time.sleep(periode / acceleration)
        yield "simulation", ligne


class Detecteur:
    """Fenêtre glissante et niveau d'alerte d'un nœud."""

    def __init__(self, modele):
        self.modele = modele
        self.fenetre = deque(maxlen=taille_fenetre)
        self.consecutives = 0
        self.normales = 0
        self.niveau = None   # None, "warning" ou "confirmed" pour l'épisode en cours
        self.n = 0           # nombre de mesures traitées

    def couper(self):
        """La série est interrompue : on repart d'une fenêtre vide."""
        self.fenetre.clear()
        self.consecutives = 0

    def traiter(self, mesure):
        """Retourne (etat, score, nouveau_niveau). nouveau_niveau vaut "warning" ou
        "confirmed" quand une alerte doit être envoyée, sinon None."""
        self.n += 1
        self.fenetre.append(mesure)

        # Attente du remplissage de la fenêtre
        if len(self.fenetre) < taille_fenetre:
            return f"remplissage ({len(self.fenetre)}/{taille_fenetre})", None, None

        # Extraction des caractéristiques et prédiction
        x = extraire(np.array(self.fenetre)).reshape(1, -1)
        score = float(self.modele.decision_function(x)[0])
        anormal = self.modele.predict(x)[0] == -1

        # Confirmation de l'anomalie
        if anormal:
            self.consecutives += 1
            self.normales = 0
        else:
            self.consecutives = 0
            self.normales += 1
            if self.normales >= fin_episode:
                self.niveau = None

        nouveau = None
        if self.consecutives >= k_confirmation and self.niveau != "confirmed":
            nouveau = self.niveau = "confirmed"
        elif self.consecutives >= k_avertissement and self.niveau is None:
            nouveau = self.niveau = "warning"

        if self.consecutives >= k_confirmation:
            etat = "ALERTE ANOMALIE"
        elif anormal:
            etat = "suspect"
        else:
            etat = "normal"
        return etat, score, nouveau


def message_alerte(niveau, temp, hum, gaz, score):
    critique = temp >= seuil_critique_temp or gaz >= seuil_critique_gaz
    debut = "Motif anormal qui commence" if niveau == "warning" else "Anomalie confirmée (le motif se poursuit)"
    seuil = "seuil critique déjà atteint" if critique else "avant tout seuil critique"
    return (f"{debut}, {seuil} : temp={temp:.1f}°C, hum={hum:.1f}%, gaz={gaz:.0f} "
            f"(score Isolation Forest {score:.3f})")


def main():

    parseur = argparse.ArgumentParser(
        description="Détection d'anomalies sur un flux de capteurs."
    )

    parseur.add_argument(
        "--source",
        default="simulation",
        choices=["simulation", "mqtt"],
        help="simulation : données générées ; mqtt : télémétrie réelle des nœuds."
    )

    parseur.add_argument(
        "--scenario",
        default="derive",
        choices=["normal", "derive", "pic_gaz"],
        help="Scénario de simulation."
    )

    parseur.add_argument(
        "--acceleration",
        type=float,
        default=20,
        help="Facteur d'accélération de la simulation."
    )

    parseur.add_argument(
        "--envoyer",
        action="store_true",
        help="Envoyer aussi les alertes de la simulation au backend (toujours fait avec --source mqtt)."
    )

    args = parseur.parse_args()


    if not fichier_modele.exists():
        print("\nERREUR : le fichier du modèle est introuvable.")
        print(f"Chemin recherché : {fichier_modele}")
        print()
        print("Si le modèle n'a pas encore été créé,")
        print("il faut d'abord exécuter le script d'entraînement.")
        return


    # Chargement du modèle

    try:
        modele = joblib.load(fichier_modele)
    except Exception as e:
        print("\nERREUR lors du chargement du modèle :")
        print(e)
        return

    client = None
    if args.source == "mqtt" or args.envoyer:
        from alertes import ClientAlertes
        client = ClientAlertes()

    if args.source == "mqtt":
        from flux_mqtt import flux_mqtt, COUPURE
        flux = flux_mqtt()
        print("Source : télémétrie MQTT réelle")
    else:
        COUPURE = None
        flux = flux_fictif(args.scenario, args.acceleration)
        print(f"Scénario : {args.scenario} | accélération : {args.acceleration}x")

    print(f"Modèle chargé : {fichier_modele}")
    print()

    detecteurs = {}

    # Traitement du flux

    for appareil, mesure in flux:
        detecteur = detecteurs.setdefault(appareil, Detecteur(modele))

        if mesure is COUPURE:
            if len(detecteur.fenetre):
                print(f"[{appareil}] coupure de la série (mesure invalide, préchauffage, perte ou redémarrage) : fenêtre vidée")
            detecteur.couper()
            continue

        etat, score, nouveau = detecteur.traiter(mesure)
        temp, hum, gaz = mesure
        t = detecteur.n * periode

        # Affichage
        if score is None:
            print(f"[{appareil}] [{t:5d}s] {etat}", end="\r")
        elif detecteur.n % 15 == 0 or nouveau:
            print(
                f"[{appareil}] [{t:5d}s] "
                f"{etat:15} | "
                f"temp={temp:.1f}°C, "
                f"hum={hum:.1f}%, "
                f"gaz={gaz:.1f} | "
                f"score={score:.3f}"
            )

        # Déclenchement de l'alerte (un niveau envoyé une seule fois par épisode)
        if nouveau:
            message = message_alerte(nouveau, temp, hum, gaz, score)
            print(f"\n>>> {nouveau.upper()} [{appareil}] {message}\n")
            if client and appareil != "simulation":
                client.envoyer(nouveau, "anomaly", appareil, message,
                               donnees={"score": round(score, 4), "consecutive_windows": detecteur.consecutives,
                                        "temp_c": round(float(temp), 1), "hum_pct": round(float(hum), 1), "gas_mv": round(float(gaz), 1)})
            elif client:
                client.envoyer(nouveau, "anomaly", os.environ.get("DEVICE_ID", ""), f"[simulation {args.scenario}] {message}",
                               donnees={"score": round(score, 4), "simulation": True})


if __name__ == "__main__":
    main()
