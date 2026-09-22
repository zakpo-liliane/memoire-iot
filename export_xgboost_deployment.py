"""Export XGBoost pour l'application WATCHER.

Dans la derniere cellule de memoire_finale_iot.ipynb, executez :
    %run -i export_xgboost_deployment.py
"""

from pathlib import Path
import json

import joblib
from xgboost import XGBClassifier

required = ("X_train_39", "y_train", "scaler", "encoder_y", "xgb_params", "PROJET_DIR")
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(
        "Variables notebook absentes : " + ", ".join(missing) +
        ". Executez d'abord les cellules de preparation et XGBoost."
    )

features = X_train_39
if not hasattr(features, "columns"):
    raise TypeError("X_train_39 doit etre un DataFrame avec les 39 noms de variables.")

deployment_dir = Path(PROJET_DIR) / "deployment"
deployment_dir.mkdir(parents=True, exist_ok=True)
model = XGBClassifier(**xgb_params)
model.set_params(device="cpu", tree_method="hist")
model.fit(features, y_train)

joblib.dump(model, deployment_dir / "model_xgboost.joblib")
joblib.dump(scaler, deployment_dir / "scaler.joblib")
joblib.dump(encoder_y, deployment_dir / "label_encoder.joblib")
(deployment_dir / "feature_names.json").write_text(
    json.dumps(list(features.columns), ensure_ascii=False), encoding="utf-8"
)
(deployment_dir / "deployment_manifest.json").write_text(
    json.dumps(
        {
            "model": "xgboost",
            "feature_count": len(features.columns),
            "validation_status": "PENDING",
            "message": "Execution du test independant requise avant activation dans WATCHER.",
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(f"Export termine : {deployment_dir}")
print("Executez ensuite evaluate_deployment.py sur le jeu de test independant.")
