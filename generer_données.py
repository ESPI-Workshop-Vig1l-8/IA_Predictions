
import numpy as np
import os
import pandas as pd

periode = 2

colonnes = ['temperature', 'humidité', 'gaz']


# Génère des séries de données pour l'apprentissage du modèle
def generer_serie(n, scenario="normal", debut_anomalie=None, graine=None):

    rng = np.random.default_rng(graine)

    t = np.arange(n) * periode

    phase = rng.uniform(0, 2 * np.pi)

    # Variation de température autour de 24°C
    temp = (
        24
        + 1.5 * np.sin(2 * np.pi * t / (6 * 3600) + phase)
        + rng.normal(0, 0.15, n)
    )

    # Humidité anti-corrélée à la température
    hum = (
        45
        - 1.2 * (temp - 24)
        + rng.normal(0, 0.8, n)
    )

    # Niveau normal de gaz
    gaz = 250 + rng.normal(0, 4, n)

    # Définition des scénarios avec anomalie
    if scenario != "normal":

        if debut_anomalie is None:
            debut_anomalie = n // 3

        minutes = np.clip(
            t - debut_anomalie * periode,
            0,
            None
        ) / 60

        if scenario == "derive":

            temp += 0.4 * minutes
            gaz += 6 * minutes
            hum -= 0.5 * minutes

        elif scenario == "pic_gaz":

            gaz += np.clip(minutes * 10, 0, 1) * 250

        else:
            raise ValueError(f"Scenario inconnu : {scenario}")

    return pd.DataFrame({
        'temperature': temp,
        'humidité': hum,
        'gaz': gaz
    })


if __name__ == "__main__":

    os.makedirs("donnees", exist_ok=True)

    n_24h = 24 * 3600 // periode

    donnees = generer_serie(
        n_24h,
        "normal",
        graine=0
    )

    donnees.round(2).to_csv(
        "donnees/normal.csv",
        index=False
    )

    print(
        f"donnees/normal.csv écrit ({n_24h} mesures)"
    )
