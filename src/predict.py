"""Projet 2 - Prediction avec le modele entraine.

Usage : python src/predict.py chemin/vers/flux.csv
ou, depuis Python : from predict import predict
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

MODELE = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
COLONNES = [
    "duration", "protocol", "dst_port", "src_bytes", "dst_bytes",
    "nb_packets", "nb_failed_logins", "nb_conn_same_host", "syn_rate",
]

_modele = None


def _charger_modele():
    global _modele
    if _modele is None:
        _modele = joblib.load(MODELE)
    return _modele


def predict(df: pd.DataFrame) -> list:
    """Retourne une liste de 0 (normal) ou 1 (attaque), une valeur par ligne de df."""
    X = df.reindex(columns=COLONNES)  # ignore 'attaque' si presente, fixe l'ordre
    return [int(v) for v in _charger_modele().predict(X)]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage : python src/predict.py flux.csv")
    print(predict(pd.read_csv(sys.argv[1])))
