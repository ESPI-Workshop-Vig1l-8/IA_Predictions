"""
Envoi des alertes au backend : POST /api/v1/alerts (jeton service).

Variables : BACKEND_URL (défaut http://backend:5000 dans Docker ; depuis le PC
serveur, passer par le dashboard : http://127.0.0.1:10443) et API_SERVICE_TOKEN.
"""

import os

import requests


class ClientAlertes:
    def __init__(self, source="ia-prediction"):
        self.url = os.environ.get("BACKEND_URL", "http://backend:5000").rstrip("/")
        self.jeton = os.environ.get("API_SERVICE_TOKEN", "")
        self.source = source
        if not self.jeton:
            print("ATTENTION : API_SERVICE_TOKEN absent, les alertes seront refusées par le backend.")

    def envoyer(self, niveau, type_alerte, appareil, details="", confiance=None, donnees=None):
        """niveau : "warning" (début d'un motif anormal) ou "confirmed" (le motif se poursuit,
        la LED environnement du nœud clignote). Retourne True si l'alerte est enregistrée."""
        corps = {"source": self.source, "type": type_alerte, "level": niveau,
                 "device_id": appareil, "details": details[:500]}
        if confiance is not None:
            corps["confidence"] = round(min(max(confiance, 0.0), 1.0), 3)
        if donnees:
            corps["data"] = donnees
        try:
            reponse = requests.post(f"{self.url}/api/v1/alerts", json=corps, timeout=5,
                                    headers={"Authorization": f"Bearer {self.jeton}"})
        except requests.RequestException as erreur:
            print(f"!!! Alerte non envoyée (backend injoignable sur {self.url}) : {erreur}")
            return False
        if reponse.status_code != 201:
            print(f"!!! Alerte refusée par le backend ({reponse.status_code}) : {reponse.text[:200]}")
            return False
        print(f">>> Alerte {niveau} envoyée : {reponse.json().get('action')}")
        return True
