"""
Export des mesures CouchDB vers un CSV d'entraînement (format de generer_données.py).

    python exporter_couchdb.py --appareil VIG1L-8-NODE04 --debut 2026-10-06T08:00 --fin 2026-10-07T08:00

Connexion : variables COUCHDB_URL, COUCHDB_USER, COUCHDB_PASSWORD (fichier .env accepté).
L'utilisateur "ia" est en lecture seule.

Seules les mesures "normales" sont gardées, pour que l'Isolation Forest n'apprenne
pas des anomalies comme normales :
  - DHT22 en erreur ou valeurs manquantes        -> rejeté
  - MQ-2 en préchauffage (status.gas_warm false)  -> rejeté
  - période couverte par une annotation (tests)   -> rejeté
Les mesures restantes sont découpées en segments continus (pas de message perdu,
pas de redémarrage, pas de trou de temps). Une fenêtre ne doit jamais chevaucher
deux segments : la colonne "segment" du CSV permet de les séparer.
"""

import argparse
import json
import os
from datetime import datetime, timedelta

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

periode = 2               # secondes entre deux mesures (comme generer_données.py)
taille_fenetre = 60       # comme caracteristiques.py
colonnes = {"temp_c": "temperature", "hum_pct": "humidite", "gas_mv": "gaz"}


def connexion():
    session = requests.Session()
    session.auth = (os.environ.get("COUCHDB_USER", "ia"), os.environ.get("COUCHDB_PASSWORD", ""))
    url = os.environ.get("COUCHDB_URL", "http://127.0.0.1:5984").rstrip("/")
    return session, url


def lire_plage(session, url, base, debut_cle, fin_cle, par_page=2000):
    """Lit tous les documents dont l'_id est entre debut_cle et fin_cle (pagination)."""
    docs = []
    params = {"startkey": json.dumps(debut_cle), "endkey": json.dumps(fin_cle),
              "include_docs": "true", "limit": par_page}
    while True:
        reponse = session.get(f"{url}/{base}/_all_docs", params=params, timeout=30)
        reponse.raise_for_status()
        lignes = reponse.json()["rows"]
        docs += [ligne["doc"] for ligne in lignes]
        if len(lignes) < par_page:
            return docs
        # page suivante : on repart du dernier _id lu, sans le relire
        params["startkey"] = json.dumps(lignes[-1]["id"])
        params["skip"] = 1


def lire_mesures(session, url, appareil, debut_ms, fin_ms):
    return lire_plage(session, url, "telemetry", f"{appareil}:{debut_ms:013d}", f"{appareil}:{fin_ms:013d}")


def lire_annotations(session, url, appareil):
    docs = lire_plage(session, url, "events", "annotation:", "annotation:￰")
    return [(d["start"], d["end"]) for d in docs
            if d.get("type") == "annotation" and d.get("device_id") in (appareil, None)]


def convertir(docs, annotations, garder_annotations=False, taille_min=taille_fenetre):
    """Documents CouchDB -> DataFrame (horodatage, segment, temperature, humidite, gaz)."""
    if not docs:
        return pd.DataFrame(columns=["horodatage", "segment", *colonnes.values()]), {}

    df = pd.json_normalize(docs)
    for col in ["temp_c", "hum_pct", "gas_mv", "status.dht", "status.gas_warm"]:
        if col not in df:
            df[col] = None
    df = df.sort_values("received_at").reset_index(drop=True)
    rejets = {}

    # Segments calculés sur toutes les mesures reçues, avant filtrage
    nouveau = (
        (df["seq"].diff() != 1)                               # message perdu, ou redémarrage
        | (df["uptime_ms"].diff() < 0)                        # redémarrage
        | (df["received_at"].diff() > 3 * periode * 1000)     # trou de temps
    )
    df["segment"] = nouveau.cumsum()

    valide = (df["status.dht"] == "ok") & df[list(colonnes)].notna().all(axis=1)
    rejets["capteur en erreur / valeur manquante"] = int((~valide).sum())
    chaud = df["status.gas_warm"] == True  # noqa: E712 (None doit compter comme faux)
    rejets["préchauffage MQ-2"] = int((valide & ~chaud).sum())
    garde = valide & chaud

    if not garder_annotations:
        annote = pd.Series(False, index=df.index)
        for debut, fin in annotations:
            annote |= df["received_at"].between(debut, fin)
        rejets["période annotée (test)"] = int((garde & annote).sum())
        garde &= ~annote

    # Une mesure rejetée coupe le segment : on renumérote après filtrage
    df["segment"] = (df["segment"] + (~garde).cumsum()).where(garde)
    df = df[garde].copy()
    df["segment"] = pd.factorize(df["segment"])[0]

    taille = df.groupby("segment")["segment"].transform("size")
    rejets[f"segment trop court (< {taille_min} mesures)"] = int((taille < taille_min).sum())
    df = df[taille >= taille_min].copy()
    df["segment"] = pd.factorize(df["segment"])[0]

    df["horodatage"] = pd.to_datetime(df["received_at"], unit="ms", utc=True)
    df = df.rename(columns=colonnes)[["horodatage", "segment", *colonnes.values()]]
    return df.reset_index(drop=True), rejets


def date_ms(texte):
    """Date ISO (heure locale si pas de fuseau) -> epoch en millisecondes."""
    return int(datetime.fromisoformat(texte).astimezone().timestamp() * 1000)


def main():
    maintenant = datetime.now()
    parser = argparse.ArgumentParser(description="Export CouchDB -> CSV d'entraînement")
    parser.add_argument("--appareil", default="VIG1L-8-NODE04")
    parser.add_argument("--debut", default=(maintenant - timedelta(days=1)).isoformat(timespec="minutes"),
                        help="date ISO, heure locale (défaut : il y a 24 h)")
    parser.add_argument("--fin", default=maintenant.isoformat(timespec="minutes"),
                        help="date ISO, heure locale (défaut : maintenant)")
    parser.add_argument("--sortie", default="donnees/normal.csv")
    parser.add_argument("--garder-annotations", action="store_true",
                        help="garder les périodes de test (pour évaluer le modèle)")
    parser.add_argument("--taille-min", type=int, default=taille_fenetre,
                        help="taille minimale d'un segment (défaut : une fenêtre)")
    args = parser.parse_args()

    session, url = connexion()
    docs = lire_mesures(session, url, args.appareil, date_ms(args.debut), date_ms(args.fin))
    annotations = lire_annotations(session, url, args.appareil)
    donnees, rejets = convertir(docs, annotations, args.garder_annotations, args.taille_min)

    print(f"{len(docs)} mesures lues pour {args.appareil} ({args.debut} -> {args.fin})")
    for raison, nombre in rejets.items():
        if nombre:
            print(f"  - {nombre} rejetées : {raison}")
    print(f"{len(donnees)} mesures gardées en {donnees['segment'].nunique()} segment(s)")

    if donnees.empty:
        print("Rien à écrire.")
        return
    os.makedirs(os.path.dirname(args.sortie) or ".", exist_ok=True)
    donnees.round({col: 2 for col in colonnes.values()}).to_csv(args.sortie, index=False)
    print(f"{args.sortie} écrit")


if __name__ == "__main__":
    main()
