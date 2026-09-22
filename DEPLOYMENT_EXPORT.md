# Export du modele pour l'application

Les fichiers de `artifacts/bilstm/fold_*` sont des modeles de validation
croisee : ils ne doivent pas etre charges directement par l'application.
Apres avoir choisi un des cinq modeles a deployer, executez une cellule de
reentrainement sur toutes les donnees puis exportez ses objets de preparation.

La methode la plus simple est d'ouvrir `memoire_finale_iot.ipynb`, d'executer
la preparation et l'entrainement XGBoost, puis d'ajouter une derniere cellule :

```python
%run -i export_xgboost_deployment.py
```

Le script `export_xgboost_deployment.py` cree automatiquement les quatre
fichiers requis dans `deployment/` : le modele, le scaler, l'encodeur des
classes, les 39 noms de variables et un manifeste `PENDING`.

Ensuite, utilisez un CSV de test qui n'a participé ni à l'entraînement ni au
choix des paramètres, puis lancez :

```powershell
python evaluate_deployment.py --test-csv chemin\vers\test_independant.csv --target label2
```

Cette étape produit `independent_test_metrics.json`, marque le déploiement
comme `VALIDATED` et active la prédiction dans l'application.

Equivalent manuel pour XGBoost :

```python
from pathlib import Path
import json
import joblib
from xgboost import XGBClassifier

deployment_dir = PROJET_DIR / "deployment"
deployment_dir.mkdir(exist_ok=True)

# xgb_params, X_train_39, y_train, scaler et encoder_y sont definis dans le notebook.
model_deployment = XGBClassifier(**xgb_params)
model_deployment.fit(X_train_39, y_train)

joblib.dump(model_deployment, deployment_dir / "model_xgboost.joblib")
joblib.dump(scaler, deployment_dir / "scaler.joblib")
joblib.dump(encoder_y, deployment_dir / "label_encoder.joblib")
(deployment_dir / "feature_names.json").write_text(
    json.dumps(list(X_train_39.columns)), encoding="utf-8"
)
print("Export termine :", deployment_dir)
```

Pour Random Forest, remplacez le bloc de creation et d'entrainement par
`RandomForestClassifier(**rf_params)` et appelez le fichier
`model_random_forest.joblib`. L'application detecte automatiquement le fichier
exporte et active alors la prediction CSV.
