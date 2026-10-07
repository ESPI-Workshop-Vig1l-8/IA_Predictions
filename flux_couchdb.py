"""
Flux réel : nouvelles mesures lues dans CouchDB (flux _changes de la base telemetry).

Seul le backend écrit dans cette base, et seulement des messages qu'il a validés
(format, device_id cohérent avec le topic) : l'IA ne lit donc que des données sûres,
avec le compte "ia" en lecture seule (variables COUCHDB_URL, COUCHDB_USER,
COUCHDB_PASSWORD, comme exporter_couchdb.py). Le backend écrit par lots toutes les
2 s : les mesures arrivent avec quelques secondes de retard, négligeable devant
des fenêtres de 2 minutes.

À la première mesure d'un nœud, ses 2 dernières minutes sont relues en base : la
détection démarre tout de suite, même après un redémarrage du service. Ces mesures
sont marquées "historique" : elles remplissent la fenêtre sans relancer d'alerte
(déjà envoyée avant le redémarrage).
"""

import time

import numpy as np
import requests

from caracteristiques import taille_fenetre
from exporter_couchdb import connexion, lire_mesures
from generer_données import periode

COUPURE = None  # la série est interrompue : la fenêtre glissante doit être vidée
ATTENTE_S = 30  # durée max d'une requête longpoll sans nouvelle mesure
# Le flux _changes n'est pas ordonné entre les shards d'une base CouchDB : les mesures
# sont gardées REORDRE_MS avant d'être traitées dans l'ordre de received_at.
REORDRE_MS = 3000


def _mesure(doc):
    """[température, humidité, gaz], ou None si la mesure n'est pas utilisable
    (DHT22 en erreur, MQ-2 en préchauffage) : mêmes règles qu'à l'entraînement."""
    statut = doc.get("status") or {}
    valeurs = [doc.get("temp_c"), doc.get("hum_pct"), doc.get("gas_mv")]
    if statut.get("dht") != "ok" or not statut.get("gas_warm") or any(v is None for v in valeurs):
        return None
    return np.array(valeurs, dtype=float)


class _Suivi:
    """Détecte les coupures de la série d'un nœud : trou dans seq (message perdu)
    ou uptime qui recule (redémarrage)."""

    def __init__(self):
        self.dernier = {}

    def traiter(self, doc, historique=False):
        appareil = doc["device_id"]
        seq, uptime, recu = doc.get("seq"), doc.get("uptime_ms") or 0, doc.get("received_at") or 0
        precedent = self.dernier.get(appareil)
        if precedent and recu <= precedent[2]:
            return  # déjà vu (préchargement puis flux) ; received_at ne recule jamais
        self.dernier[appareil] = (seq, uptime, recu)
        if precedent and (seq != precedent[0] + 1 or uptime < precedent[1]):
            yield appareil, COUPURE, historique
        mesure = _mesure(doc)
        yield appareil, mesure if mesure is not None else COUPURE, historique


def flux_couchdb():
    """Génère (appareil, mesure, historique) avec mesure = [température, humidité, gaz],
    ou COUPURE quand la série est interrompue ; historique vaut True pour les mesures
    préchargées (pas d'alerte)."""
    session, url = connexion()
    suivi = _Suivi()
    since = "now"
    en_attente = []
    print(f"[CouchDB] Lecture des nouvelles mesures sur {url}/telemetry (utilisateur {session.auth[0]})")

    while True:
        try:
            attente_ms = 1000 if en_attente else ATTENTE_S * 1000
            reponse = session.get(f"{url}/telemetry/_changes", timeout=ATTENTE_S + 10, params={
                "feed": "longpoll", "since": since, "include_docs": "true",
                "timeout": attente_ms, "limit": 500,
            })
            if reponse.status_code == 401:
                raise SystemExit(f"Accès refusé par {url} : vérifier COUCHDB_USER / COUCHDB_PASSWORD")
            reponse.raise_for_status()
        except requests.RequestException as erreur:
            print(f"[CouchDB] Injoignable ({erreur}), nouvel essai dans 5 s")
            time.sleep(5)
            continue

        resultat = reponse.json()
        since = resultat["last_seq"]
        en_attente += [c["doc"] for c in resultat["results"]
                       if not c["id"].startswith("_design/") and not c.get("deleted") and c.get("doc", {}).get("device_id")]
        en_attente.sort(key=lambda d: d.get("received_at", 0))
        limite = time.time() * 1000 - REORDRE_MS
        prets = [d for d in en_attente if d.get("received_at", 0) <= limite]
        en_attente = en_attente[len(prets):]

        for doc in prets:
            appareil = doc["device_id"]
            if appareil not in suivi.dernier:
                # premières mesures de ce nœud : on relit ses 2 dernières minutes
                fin = doc["received_at"]
                debut = fin - (taille_fenetre + 5) * periode * 1000
                try:
                    historique = lire_mesures(session, url, appareil, debut, fin - 1)
                except requests.RequestException:
                    historique = []
                if historique:
                    print(f"[CouchDB] {appareil} : {len(historique)} mesures récentes préchargées")
                for ancien in sorted(historique, key=lambda d: d.get("received_at", 0)):
                    yield from suivi.traiter(ancien, historique=True)
            yield from suivi.traiter(doc)
