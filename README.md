# Cyber threat detection using ML methodologies

An experimental NetFlow intrusion-detection benchmark and FastAPI interface. The original Google Colab notebook is preserved, without cell outputs, in `notebooks/`. A separate reproducibility script benchmarks three model families on three public datasets and exports a small decision tree for the web application.

> Research prototype only. It is not a real-time monitoring system, a calibrated risk model, or a security control. Predictions and random-split benchmark scores should not be used to make security decisions.

## Run the app

Python 3.13 was used for the verified local run.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000/>. Interactive API documentation is at `/docs`. The application uses the included JSON decision-tree model; no training dependencies or external services are required to run the app. It processes a single manually entered flow per request, does not capture network traffic and does not store input records in the app. Infrastructure request logs can still exist.

## API

- `GET /api/health`: process health, not a model readiness check.
- `GET /api/model`: selected model and feature metadata.
- `GET /api/results`: measured benchmark results and methodology.
- `POST /api/predict`: eight numeric features, returning label, class and the attack class's tree-leaf fraction. Label 1 is attack; 0 is benign. The fraction is not calibrated. Invalid, missing or extra fields return 422.

```bash
curl -s http://127.0.0.1:8000/api/predict -H 'Content-Type: application/json' -d '{"PROTOCOL":6,"TCP_FLAGS":24,"L7_PROTO":7,"IN_BYTES":420,"OUT_BYTES":320,"IN_PKTS":5,"OUT_PKTS":4,"FLOW_DURATION_MILLISECONDS":200}'
```

The example is synthetic; it is not a known attack or benign flow.

## Reproduce the benchmark

Datasets: [NF-UNSW-NB15](https://www.kaggle.com/datasets/dhoogla/nfunswnb15), [NF-BoT-IoT](https://www.kaggle.com/datasets/dhoogla/nfbotiot), [NF-ToN-IoT](https://www.kaggle.com/datasets/dhoogla/nftoniot). Check their terms before using or redistributing them. Data is not included in the repository. Download and extract the corresponding parquet files into `data/` (or another directory):

```bash
pip install -r requirements-training.txt
python -m ml.train --data-dir data --sample-size 15000 --folds 3 --trees 60
```

The training script updates `ml/results.json` and `backend/models/decision_tree.json`. It uses a stratified 15,000-row sample per dataset, an 80/20 held-out split, three-fold training cross-validation with scaling and SMOTE **inside** each fold, and random seed 42. The 60-tree random forest, decision tree, and 60-tree XGBoost models are compared with no hyperparameter tuning. The application serves the Decision Tree trained on the NF-UNSW-NB15 sample, not the forest or XGBoost models. It verifies the portable tree against sklearn predictions during export. See `ml/results.json` for exact counts, metrics, confusion matrices and dependency versions.

### Important differences from the Colab

The original notebook samples 150,000 rows for NF-UNSW-NB15 and NF-ToN-IoT, uses the full NF-BoT-IoT dataset, uses 100 estimators, five-fold scores and an extensive Random Forest grid search. Its saved output is incomplete after the NF-BoT-IoT tuning begins. Its cross-validation happens after scaling and SMOTE have been applied to the full training split, which can leak information between CV folds. The smaller benchmark above uses a fold-local pipeline and does **not** reproduce the notebook's original metrics or hyperparameter search. These are random record splits, not time-based or cross-dataset tests; duplicate or highly similar flows may occur across splits. In particular, the NF-BoT-IoT test sample contains only 70 benign flows, so its false-alarm estimate is unstable. Results are not evidence of performance on future traffic, unseen networks or new attack types.

## Tests and deployment

```bash
pip install -r requirements-test.txt
pytest -q
```

See `vercel.md` for deployment steps and `02_SYSTEM_ARCHITECTURE.md` for module responsibilities.
