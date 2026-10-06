import numpy as np 
from generer_données import periode

taille_fenetre =  60

def extraire (fenetre) : 
    f = np.asarray(fenetre, dtype = float)
    t = np.arange(len(f)) * periode
    t_c = t - t.mean()

# pente de la droite de régression ( unité par minute)
    pente = (t_c[:, None] * (f - f.mean(axis=0))).sum(axis=0) / (t_c ** 2).sum() * 60

    if f[:, 0].std() > 0 and f[:, 2].std() > 0:
        corr = np.corrcoef(f[:, 0], f[:, 2])[0, 1]
    else:
        corr = 0.0

    return np.concatenate([f.mean(axis=0), f.std(axis=0), pente, [corr]])

def fenetres (mesures, taille = taille_fenetre, pas = 1) : 
    mesures = np.asarray(mesures, dtype = float)
    fins = np.arange(taille - 1, len(mesures), pas)
    X = np.array([extraire(mesures[fin - taille + 1: fin + 1]) for fin in fins])

    return X, fins

# confirmation d'anomalie, si 3 fenetres consécutives sont anormales, on confirme l'anomalie
def confirmer (drapeaux, k=3) : 
    sortie = np.zeros(len(drapeaux), dtype = bool)
    serie = 0
    for i, d in enumerate (drapeaux) : 
        serie = serie + 1 if d else 0
        sortie [i] = serie >= k
    return sortie