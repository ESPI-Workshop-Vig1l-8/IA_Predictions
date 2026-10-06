import argparse
import time
from collections import deque

import joblib
import numpy as np

from generer_données import colonnes, periode, generer_serie
from caracteristiques import taille_fenetre, extraire