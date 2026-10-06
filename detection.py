
import argparse
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np

from generer_données import colonnes, periode, generer_serie
from caracteristiques import taille_fenetre, extraire


# Dossier dans lequel se trouve detection.py
DOSSIER_PROJET = Path(__file__).resolve().parent

# Chemin complet vers le modèle
fichier_modele = DOSSIER_PROJET / "modele_anomalies.joblib"

k_confirmation = 5

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

    for i, ligne in enumerate(serie[colonnes].to_numpy()):
        time.sleep(periode / acceleration)
        yield i, ligne



def main():

  

    parseur = argparse.ArgumentParser(
        description="Détection d'anomalies sur un flux de capteurs."
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

    args = parseur.parse_args()


    if not fichier_modele.exists():
        print("\nERREUR : le fichier du modèle est introuvable.")
        print(f"Chemin recherché : {fichier_modele}")
        print()
        print("Vérifie que le fichier suivant existe :")
        print(f"    {DOSSIER_PROJET / 'modeles_anomalies.joblib'}")
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

   
    # Initialisation de la fenêtre glissante
  

    fenetre = deque(maxlen=taille_fenetre)

    consecutives = 0
    alerte_deja_levee = False

    print(
        f"Scénario : {args.scenario} | "
        f"accélération : {args.acceleration}x"
    )

    print(f"Modèle chargé : {fichier_modele}")
    print()

  
    # Traitement du flux


    for i, mesure in flux_fictif(
        args.scenario,
        args.acceleration
    ):

        # Ajout de la mesure dans la fenêtre glissante
        fenetre.append(mesure)

        temp, hum, gaz = mesure

        # Attente du remplissage de la fenêtre


        if len(fenetre) < taille_fenetre:

            print(
                f"[{i * periode:5d}s] "
                f"remplissage de la fenêtre "
                f"({len(fenetre)}/{taille_fenetre})",
                end="\r"
            )

            continue

       
        # Extraction des caractéristiques
       

        x = extraire(
            np.array(fenetre)
        ).reshape(1, -1)

       
        # Prédiction


        score = float(
            modele.decision_function(x)[0]
        )

        anormal = modele.predict(x)[0] == -1

      
        # Confirmation de l'anomalie
       

        if anormal:
            consecutives += 1
        else:
            consecutives = 0

        alerte = consecutives >= k_confirmation

        if alerte:
            etat = "ALERTE ANOMALIE"
        elif anormal:
            etat = "suspect"
        else:
            etat = "normal"

     
        # Affichage
        

        if i % 15 == 0 or (alerte and not alerte_deja_levee):

            print(
                f"[{i * periode:5d}s] "
                f"{etat:15} | "
                f"temp={temp:.1f}°C, "
                f"hum={hum:.1f}%, "
                f"gaz={gaz:.1f} | "
                f"score={score:.3f}"
            )

       
        # Déclenchement de l'alerte
      

        if alerte and not alerte_deja_levee:

            alerte_deja_levee = True

            critique = (
                temp >= seuil_critique_temp
                or gaz >= seuil_critique_gaz
            )

            if critique:
                message = "seuil critique déjà atteint"
            else:
                message = "avant tout seuil critique"

            print(
                f"\n>>> Anomalie prédite à "
                f"{i * periode}s "
                f"({message})\n"
            )

            # Ici :
            # envoyer l'alerte sur le backend en on point
           

        # Réinitialisation de l'état d'alerte
       

        if not alerte:
            alerte_deja_levee = False



if __name__ == "__main__":
    main()

