"""
Flux réel : télémétrie des nœuds lue sur le broker MQTT (vigil8/<id>/telemetry).

Variables : MQTT_BROKER (tcp://mqtt-broker:1883 dans Docker, ou
ssl://192.168.10.1:18883 depuis le PC serveur avec MQTT_CA_FILE=certs/ca.crt),
MQTT_USERNAME (ia-prediction, lecture seule), MQTT_PASSWORD.
"""

import json
import importlib
import os
import queue
from urllib.parse import urlparse

import numpy as np
mqtt = importlib.import_module("paho.mqtt.client")

COUPURE = None  # la série est interrompue : la fenêtre glissante doit être vidée


def flux_mqtt():
    """Génère (appareil, mesure) avec mesure = [température, humidité, gaz], ou
    (appareil, COUPURE) quand la mesure n'est pas utilisable (DHT22 en erreur,
    MQ-2 en préchauffage) ou qu'un message manque (trou dans seq, redémarrage) :
    mêmes règles que pour l'entraînement (exporter_couchdb.py)."""
    url = urlparse(os.environ.get("MQTT_BROKER", "tcp://mqtt-broker:1883"))
    tls = url.scheme in ("ssl", "mqtts", "tls")
    messages = queue.Queue(maxsize=1000)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=os.environ.get("MQTT_CLIENT_ID", "ia-prediction"))
    client.username_pw_set(os.environ.get("MQTT_USERNAME", "ia-prediction"), os.environ.get("MQTT_PASSWORD", ""))
    if tls:
        client.tls_set(ca_certs=os.environ.get("MQTT_CA_FILE"))

    def on_connect(c, _userdata, _flags, reason_code, _properties):
        if reason_code.is_failure:
            print(f"[MQTT] Connexion refusée : {reason_code} (identifiants ? ACL ?)")
            return
        print("[MQTT] Connecté, abonné à vigil8/+/telemetry")
        c.subscribe("vigil8/+/telemetry", qos=0)

    def on_message(_c, _userdata, msg):
        try:
            messages.put_nowait((msg.topic.split("/")[1], json.loads(msg.payload)))
        except (ValueError, IndexError, queue.Full):
            pass

    client.on_connect = on_connect
    client.on_message = on_message
    client.reconnect_delay_set(1, 30)
    client.connect_async(url.hostname, url.port or (8883 if tls else 1883))
    client.loop_start()
    print(f"[MQTT] Connexion à {url.hostname}:{url.port or (8883 if tls else 1883)} ({'TLS' if tls else 'interne'})...")

    dernier = {}
    while True:
        appareil, doc = messages.get()
        if doc.get("device_id") != appareil:
            continue  # le backend rejette déjà ces messages

        seq, uptime = doc.get("seq"), doc.get("uptime_ms")
        precedent = dernier.get(appareil)
        dernier[appareil] = (seq, uptime)
        if precedent and (seq != precedent[0] + 1 or (uptime or 0) < (precedent[1] or 0)):
            yield appareil, COUPURE

        statut = doc.get("status") or {}
        valeurs = [doc.get("temp_c"), doc.get("hum_pct"), doc.get("gas_mv")]
        if statut.get("dht") != "ok" or not statut.get("gas_warm") or any(v is None for v in valeurs):
            yield appareil, COUPURE
            continue
        yield appareil, np.array(valeurs, dtype=float)
