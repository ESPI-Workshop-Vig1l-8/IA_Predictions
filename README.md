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
