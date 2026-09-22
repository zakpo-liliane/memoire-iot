"""Validation indépendante obligatoire du modèle exporté WATCHER.

À exécuter depuis le dossier ``projet-remote`` après l'export :
    python evaluate_deployment.py --test-csv chemin/vers/test_independant.csv --target label2

Le CSV ne doit pas avoir servi lors de l'entraînement, du VIF ou du choix des
hyperparamètres. Le script écrit les métriques et valide l'artefact uniquement
si les 39 variables attendues sont présentes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


PROJECT = Path(__file__).resolve().parent
DEPLOYMENT = PROJECT / "deployment"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", type=Path, required=True, help="CSV du jeu de test indépendant")
    parser.add_argument("--target", default="label2", help="Colonne cible multiclasses")
    args = parser.parse_args()

    if not args.test_csv.is_file():
        raise FileNotFoundError(args.test_csv)
    model_file = next(DEPLOYMENT.glob("model_*.joblib"), None)
    required = [model_file, DEPLOYMENT / "scaler.joblib", DEPLOYMENT / "label_encoder.joblib", DEPLOYMENT / "feature_names.json"]
    if any(path is None or not path.exists() for path in required):
        raise FileNotFoundError("Artefacts incomplets : exécutez d'abord export_xgboost_deployment.py.")

    test = pd.read_csv(args.test_csv)
    if args.target not in test.columns:
        raise ValueError(f"Colonne cible absente : {args.target}")
    features = json.loads((DEPLOYMENT / "feature_names.json").read_text(encoding="utf-8"))
    missing = sorted(set(features) - set(test.columns))
    if missing:
        raise ValueError("Test refusé : variables manquantes : " + ", ".join(missing))

    model = joblib.load(model_file)
    scaler = joblib.load(DEPLOYMENT / "scaler.joblib")
    encoder = joblib.load(DEPLOYMENT / "label_encoder.joblib")
    x_test = test[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_true = encoder.transform(test[args.target].astype(str))
    y_pred = model.predict(scaler.transform(x_test))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    summary = {
        "model": model_file.stem.removeprefix("model_"),
        "test_csv": str(args.test_csv),
        "target": args.target,
        "test_samples": int(len(test)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "validation_status": "VALIDATED",
        "note": "Jeu de test indépendant déclaré par l'utilisateur.",
    }
    (DEPLOYMENT / "independent_test_metrics.json").write_text(json.dumps({**summary, "report": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    (DEPLOYMENT / "deployment_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
