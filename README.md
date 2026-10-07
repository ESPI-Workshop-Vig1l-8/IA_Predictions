## Description
Maintenance prédictive (Isolation Forest)
L'objectif est de surveiller un flux de données provenant de capteurs et de détecter automatiquement des comportements inhabituels à partir de plusieurs mesures, la température, l'humidité et le taux de gaz.

Le système utilise un modèle de Machine Learning entraîné sur des données normales afin d'identifier des situations potentiellement anormales.

Lorsqu'une anomalie est détectée pendant plusieurs fenêtres consécutives, une alerte est déclenchée.

Le fonctionnement général du système est le suivant :
Données des capteurs ( ou générées dans le cas du test)
        │
        ▼
Lecture des données
        │
        ▼
Extraction des caractéristiques
        │
        ▼
Fenêtre glissante
        │
        ▼
Détection d'anomalie
        │
        ▼
Confirmation de plusieurs anomalies
        │
        ▼
     ALERTE
     
## Installation
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

## Configuration
Copier .env.example en .env et renseigner les valeurs.
## Données réelles (CouchDB)
`exporter_couchdb.py` récupère les mesures stockées par le backend et écrit un CSV au même format que `generer_données.py` (`temperature`, `humidite`, `gaz`) :

```
python exporter_couchdb.py --appareil VIG1L-8-NODE04 --debut 2026-10-06T08:00 --fin 2026-10-07T08:00
```

- Connexion : `COUCHDB_URL`, `COUCHDB_USER`, `COUCHDB_PASSWORD` dans `.env` (utilisateur `ia`, lecture seule). CouchDB n'écoute que sur `127.0.0.1:5984` du serveur : depuis un autre PC, passer par un tunnel SSH (`ssh -L 5984:127.0.0.1:5984 <serveur>`).
- Seules les mesures normales sont gardées : DHT22 en erreur, préchauffage du MQ-2 et périodes annotées (tests) sont retirés. `--garder-annotations` les conserve pour évaluer le modèle.
- Colonne `segment` : les mesures sont découpées en séries continues (pas de message perdu, de redémarrage ni de trou). Une fenêtre ne doit pas chevaucher deux segments.
- Sortie par défaut : `donnees/normal.csv`, utilisée par `entrainement.py`.

## Entraînement sur les vraies mesures
`entrainement.py --source couchdb` lit directement les mesures du backend (via `exporter_couchdb.py`), entraîne le modèle et l'évalue sur les vraies données :

```
python entrainement.py --source couchdb                                   # 24 dernières heures
python entrainement.py --source couchdb --debut 2026-10-06T08:00 --fin 2026-10-07T08:00
```

- Les fenêtres sont calculées **segment par segment** : une fenêtre ne chevauche jamais une coupure (message perdu, redémarrage, mesure rejetée).
- Il faut au moins 50 fenêtres (environ 20 min de mesures continues) ; plusieurs heures dont une nuit donnent un modèle plus robuste.
- Évaluation : taux de fenêtres jugées anormales sur les données normales, puis, pour chaque **période annotée** depuis le dashboard (test au briquet, souffle chaud…), si le modèle l'a détectée.
- Les données utilisées sont aussi écrites dans `donnees/normal.csv` ; `python entrainement.py` (source `csv`) réentraîne à partir de ce fichier.

## Détection en temps réel
`detection.py --source couchdb` lit les nouvelles mesures de tous les nœuds **dans CouchDB** (flux `_changes` de la base `telemetry`, compte `ia` en lecture seule) et envoie les alertes au backend (`POST /api/v1/alerts`, jeton service) :

- Seul le backend écrit en base, après validation des messages : l'IA ne lit que des données sûres et n'a aucun accès au broker MQTT.
- Le backend écrit par lots toutes les 2 s, et le flux `_changes` n'est pas ordonné entre les shards de la base : les mesures sont gardées 3 s puis traitées dans l'ordre de `received_at`. Au total, environ 5 s de retard, négligeable devant des fenêtres de 2 minutes.
- À la première mesure d'un nœud, ses 2 dernières minutes sont relues en base : la détection démarre tout de suite, même après un redémarrage du service.


| Niveau | Quand | Effet |
|---|---|---|
| `warning` | 2 fenêtres anormales consécutives : un motif anormal commence | alerte « Avertissement » sur le dashboard |
| `confirmed` | 5 fenêtres consécutives : le motif se poursuit | alerte « Confirmée », la LED environnement du nœud clignote |

- Un seul envoi par niveau et par épisode ; l'épisode se termine après 15 fenêtres normales (30 s).
- Mêmes règles qu'à l'entraînement : une mesure invalide (DHT22 en erreur, MQ-2 en préchauffage), un message perdu ou un redémarrage vide la fenêtre glissante (60 mesures, 2 min).
- Dans la stack `infra`, le service `ia-prediction` lance cette commande automatiquement (image construite depuis ce dépôt, `Dockerfile`).
- Depuis le PC serveur : renseigner `.env` (voir `.env.example.txt` : CouchDB sur `127.0.0.1:5984`, backend via le dashboard sur `10443`).
- `python detection.py --scenario derive --envoyer` envoie les alertes d'une simulation (avec `DEVICE_ID` pour faire clignoter la LED d'un nœud) : utile pour une démonstration.
- `FICHIER_MODELE` : modèle utilisé par `detection.py` et écrit par `entrainement.py` (défaut `modele_anomalies.joblib` ; dans la stack infra, `/models/modele_anomalies.joblib` sur un volume). S'il n'existe pas encore, `detection.py` utilise le modèle livré avec le dépôt.
- Réentraîner dans la stack infra, puis relancer la détection :
  ```
  docker compose run --rm ia-prediction python entrainement.py --source couchdb
  docker compose restart ia-prediction
  ```
