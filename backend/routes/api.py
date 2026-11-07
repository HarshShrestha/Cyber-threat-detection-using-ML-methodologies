import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from backend.models.schemas import Flow
from backend.services.predictor import load_model, predict_tree

router = APIRouter(prefix='/api')
ROOT = Path(__file__).resolve().parents[2]

@router.get('/health')
def health():
    return {'status': 'ok'}

@router.get('/model')
def model_info():
    try:
        model = load_model()
        return {'features': model['features'], **model['metadata']}
    except (OSError, KeyError, ValueError):
        raise HTTPException(503, 'Model unavailable. Train the model before serving.')

@router.get('/results')
def results():
    try:
        return json.loads((ROOT / 'ml' / 'results.json').read_text())
    except (OSError, ValueError):
        raise HTTPException(503, 'Benchmark results unavailable.')

@router.post('/predict')
def predict(flow: Flow):
    try:
        result = predict_tree(load_model(), flow.model_dump())
    except (OSError, KeyError, ValueError, OverflowError):
        raise HTTPException(503, 'Prediction unavailable. Check the model artifact.')
    return {**result, 'classification': 'attack' if result['label'] == 1 else 'benign',
            'disclaimer': 'Experimental research model; not a security decision or calibrated risk score.'}
