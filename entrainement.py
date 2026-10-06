import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from generer_données import colonnes, periode, generer_serie
from caracteristiques import taille_fenetre, confirmer, fenetres