# Projet 2 — Détecter des attaques réseau avec le Machine Learning

**Auteur : Goudea Nassoko** — ngoudea@gmail.com

Un modèle de détection d'intrusion (IDS) qui classe chaque flux réseau en **normal (0)** ou **attaque (1)**
(scan de ports, force brute SSH/Telnet, DoS).

## Installation et utilisation

```bash
pip install -r requirements.txt
python src/train.py                 # entraîne, compare les modèles, écrit models/model.joblib
python src/predict.py flux.csv      # prédit sur un CSV (mêmes colonnes que flux_train.csv, sans 'attaque')
```

Depuis Python :

```python
import pandas as pd
from src.predict import predict
predict(pd.read_csv("flux.csv"))   # liste de 0 / 1
```

## Méthode

1. **Exploration** : 3 000 flux, 90 valeurs manquantes (`duration`), 27,8 % d'attaques (classes déséquilibrées → F1 de la classe « attaque »).
2. **Prétraitement** (dans un `Pipeline`) : imputation par la médiane, mise à l'échelle (`StandardScaler`) et encodage one-hot de `protocol`.
3. **Modèles comparés** : régression logistique, arbre de décision, forêt aléatoire, gradient boosting.
4. **Évaluation** : jeu de validation de 25 % (stratifié) + validation croisée à 5 plis.
5. **Modèle final** : le meilleur est réentraîné sur les 3 000 flux et enregistré dans `models/model.joblib`.

## Résultats

**Nom / Prénom :** Nassoko Goudea

Jeu de validation (750 flux, dont 208 attaques) :

| Modèle | F1 (attaque) | Rappel (attaque) | Précision (attaque) | F1 validation croisée (5 plis) |
|---|---|---|---|---|
| Régression logistique | 0,919 | 0,899 | 0,940 | 0,896 |
| Arbre de décision (profondeur 6) | 0,959 | 0,957 | 0,961 | 0,935 |
| **Forêt aléatoire (300 arbres)** | **0,969** | **0,971** | **0,967** | **0,958** |
| Gradient boosting | 0,973 | 0,966 | 0,980 | 0,953 |

**Modèle retenu et pourquoi :** la **forêt aléatoire**. Elle a le meilleur F1 en validation croisée (0,958),
critère plus stable qu'un seul découpage ; en validation simple elle est quasiment à égalité avec le gradient
boosting (0,969 contre 0,973) et elle a le meilleur rappel (0,971), ce qui compte en sécurité. La régression
logistique est nettement moins bonne : la frontière entre attaque et normal n'est pas linéaire (les attaques combinent
plusieurs signaux : `syn_rate`, `nb_conn_same_host`, `nb_failed_logins`, port…).

**Analyse des erreurs (faux négatifs / faux positifs) :**

Matrice de confusion de la forêt aléatoire sur la validation :

| | Prédit normal | Prédit attaque |
|---|---|---|
| **Réel normal** | 535 (VN) | 7 (FP) |
| **Réel attaque** | 6 (FN) | 202 (VP) |

- **6 faux négatifs** : 6 attaques sur 208 ratées (rappel 97,1 %). C'est le pire cas en sécurité : une attaque non détectée
  passe inaperçue, l'intrusion, le vol de données ou la panne peuvent se poursuivre sans que personne ne réagisse.
- **7 faux positifs** : 7 flux normaux sur 542 signalés à tort (1,3 %). Ils coûtent du temps aux analystes (fausses alertes)
  mais restent moins graves qu'une attaque ratée.
- Pour réduire encore les faux négatifs, on pourrait abaisser le seuil de décision de `predict_proba`, au prix de
  plus de fausses alertes.

## Structure

```
projet-ml-Nassoko-Goudea/
├── README.md
├── requirements.txt
├── data/flux_train.csv
├── src/train.py
├── src/predict.py
└── models/model.joblib
```
