# Détection d'anomalies en continu (Isolation Forest) sur la télémétrie MQTT
FROM python:3.12-slim
WORKDIR /app
RUN useradd -r -u 10001 ia
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# modèle réentraîné (volume /models) et données d'entraînement écrits par l'utilisateur ia
RUN mkdir -p /models donnees && chown -R ia /models donnees
USER ia
ENV PYTHONUNBUFFERED=1
CMD ["python", "detection.py", "--source", "mqtt"]
