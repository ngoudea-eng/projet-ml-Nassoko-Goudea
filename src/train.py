"""Projet 2 - Entrainement d'un detecteur d'attaques reseau.

Usage (depuis la racine du depot) : python src/train.py
Compare plusieurs modeles sur un jeu de validation, garde le meilleur (F1 de la
classe "attaque"), le reentraine sur toutes les donnees et l'enregistre dans
models/model.joblib.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RACINE = Path(__file__).resolve().parent.parent
DONNEES = RACINE / "data" / "flux_train.csv"
MODELE = RACINE / "models" / "model.joblib"

CIBLE = "attaque"
COLONNES_NUM = [
    "duration", "dst_port", "src_bytes", "dst_bytes", "nb_packets",
    "nb_failed_logins", "nb_conn_same_host", "syn_rate",
]
COLONNES_CAT = ["protocol"]
GRAINE = 42


def explorer(df: pd.DataFrame) -> None:
    print("=== Exploration ===")
    print(f"Taille : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print("Valeurs manquantes :")
    print(df.isna().sum()[df.isna().sum() > 0].to_string())
    print("Repartition de la cible (proportion) :")
    print(df[CIBLE].value_counts(normalize=True).round(3).to_string())


def construire_pipeline(modele) -> Pipeline:
    """Pretraitement (imputation, echelle, encodage) + modele dans un seul Pipeline."""
    pretraitement = ColumnTransformer([
        ("num", Pipeline([
            ("imputation", SimpleImputer(strategy="median")),
            ("echelle", StandardScaler()),
        ]), COLONNES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), COLONNES_CAT),
    ])
    return Pipeline([("pretraitement", pretraitement), ("modele", modele)])


def modeles_candidats() -> dict:
    return {
        "Regression logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Arbre de decision": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=GRAINE),
        "Foret aleatoire": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=GRAINE, n_jobs=-1),
        "Gradient boosting": GradientBoostingClassifier(random_state=GRAINE),
    }


def main() -> None:
    df = pd.read_csv(DONNEES)
    explorer(df)

    X = df[COLONNES_NUM + COLONNES_CAT]
    y = df[CIBLE]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=GRAINE)

    print("\n=== Comparaison des modeles (jeu de validation 25 %) ===")
    resultats = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=GRAINE)
    for nom, modele in modeles_candidats().items():
        pipe = construire_pipeline(modele)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        f1_cv = cross_val_score(construire_pipeline(modele), X, y, cv=cv, scoring="f1").mean()
        resultats[nom] = {
            "f1": f1_score(y_val, pred),
            "rappel": recall_score(y_val, pred),
            "precision": precision_score(y_val, pred),
            "f1_cv": f1_cv,
            "pred": pred,
        }
        r = resultats[nom]
        print(f"{nom:<22} F1={r['f1']:.3f}  rappel={r['rappel']:.3f}  "
              f"precision={r['precision']:.3f}  F1 (CV 5 plis)={r['f1_cv']:.3f}")

    # Choix du meilleur modele : F1 moyen en validation croisee (plus stable
    # qu'un seul decoupage), departage par le F1 de validation.
    meilleur = max(resultats, key=lambda n: (resultats[n]["f1_cv"], resultats[n]["f1"]))
    print(f"\nModele retenu : {meilleur}")
    print("\n=== Rapport de classification du modele retenu ===")
    pred = resultats[meilleur]["pred"]
    print(classification_report(y_val, pred, target_names=["normal", "attaque"], digits=3))
    tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
    print("Matrice de confusion [[VN FP] [FN VP]] :")
    print(confusion_matrix(y_val, pred))
    print(f"Faux negatifs (attaques ratees) : {fn} sur {fn + tp} attaques")
    print(f"Faux positifs (fausses alertes) : {fp} sur {fp + tn} flux normaux")

    # Reentrainement sur toutes les donnees puis sauvegarde
    final = construire_pipeline(modeles_candidats()[meilleur])
    final.fit(X, y)
    MODELE.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODELE)
    print(f"\nModele enregistre dans {MODELE}")


if __name__ == "__main__":
    main()
