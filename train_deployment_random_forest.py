"""Entraîne et exporte le modèle de déploiement WATCHER depuis les CSV CIC-IIoT.

Le script conserve un jeu de test stratifié, distinct de l'entraînement. Par
défaut il prélève équitablement les 17 CSV afin de rester exécutable sur un PC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler


PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "Datasense-IIoT-2025" / "data" / "all_attack_benign_samples"
DEPLOYMENT = PROJECT / "deployment"
LABEL_COLUMNS = {"label_full", "label1", "label2", "label3", "label4", "label", "target"}


def load_sample(rows_per_file: int) -> pd.DataFrame:
    files = sorted(DATA.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun CSV sous {DATA}")
    frames = []
    for path in files:
        frame = pd.read_csv(path, nrows=rows_per_file, low_memory=False)
        if "label2" not in frame:
            raise ValueError(f"label2 absente : {path.name}")
        frames.append(frame)
        print(f"Charge : {path.name} ({len(frame):,} lignes)")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-file", type=int, default=8000)
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()
    if args.rows_per_file < 100 or not 0.05 <= args.test_size < 0.5:
        raise ValueError("rows-per-file >= 100 et test-size entre 0.05 et 0.50 requis.")

    data = load_sample(args.rows_per_file)
    candidates = [column for column in data.columns if column not in LABEL_COLUMNS]
    numeric = data[candidates].apply(pd.to_numeric, errors="coerce")
    usable = [column for column in candidates if numeric[column].notna().any()]
    if len(usable) < 39:
        raise ValueError(f"Seulement {len(usable)} variables numériques disponibles ; 39 requises.")
    features = usable[:39]
    x = numeric[features].fillna(0.0)
    encoder = LabelEncoder()
    y = encoder.fit_transform(data["label2"].astype(str))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, stratify=y, random_state=42
    )
    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model = RandomForestClassifier(
        n_estimators=150, max_depth=18, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=1, random_state=42,
    )
    print(f"Entraînement : {len(x_train):,} lignes, test : {len(x_test):,} lignes, 39 variables.")
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    summary = {
        "model": "random_forest",
        "feature_count": len(features),
        "training_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "sampling": f"{args.rows_per_file} premières lignes de chaque CSV source",
        "target": "label2",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "validation_status": "VALIDATED",
        "note": "Validation stratifiée sur un holdout local ; à compléter par un test temporel/externe pour une production réelle.",
    }
    DEPLOYMENT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, DEPLOYMENT / "model_random_forest.joblib")
    joblib.dump(scaler, DEPLOYMENT / "scaler.joblib")
    joblib.dump(encoder, DEPLOYMENT / "label_encoder.joblib")
    (DEPLOYMENT / "feature_names.json").write_text(json.dumps(features, ensure_ascii=False), encoding="utf-8")
    (DEPLOYMENT / "independent_test_metrics.json").write_text(json.dumps({**summary, "report": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    (DEPLOYMENT / "deployment_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
