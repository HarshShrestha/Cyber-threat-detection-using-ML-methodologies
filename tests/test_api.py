import json
from pathlib import Path
from fastapi.testclient import TestClient
from main import app
from backend.services.predictor import load_model, predict_tree

client = TestClient(app)
SAMPLE = {'PROTOCOL': 6, 'TCP_FLAGS': 24, 'L7_PROTO': 7, 'IN_BYTES': 420, 'OUT_BYTES': 320,
          'IN_PKTS': 5, 'OUT_PKTS': 4, 'FLOW_DURATION_MILLISECONDS': 200}

def test_pages_and_model_metadata():
    assert client.get('/').status_code == 200
    assert client.get('/assets/style.css').status_code == 200
    assert client.get('/api/health').json() == {'status': 'ok'}
    model = client.get('/api/model').json()
    assert model['algorithm'] == 'Decision Tree' and len(model['features']) == 8

def test_prediction_matches_tree():
    response = client.post('/api/predict', json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert {k:body[k] for k in ('label','attack_probability')} == predict_tree(load_model(), SAMPLE)
    assert body['classification'] in {'benign', 'attack'}
    assert 0 <= body['attack_probability'] <= 1

def test_rejects_invalid_requests():
    for data in [{**SAMPLE,'PROTOCOL': 256}, {**SAMPLE,'IN_BYTES': -1}, {**SAMPLE,'unrecognized': 2},
                 {k:v for k,v in SAMPLE.items() if k != 'OUT_PKTS'}]:
        assert client.post('/api/predict', json=data).status_code == 422

def test_benchmark_metrics_reconcile():
    data = client.get('/api/results').json()
    assert len(data['datasets']) == 3
    for dataset in data['datasets'].values():
        assert len(dataset['models']) == 3
        for model in dataset['models'].values():
            cm = model['confusion_matrix']; tn,fp,fn,tp = (cm[k] for k in ('tn','fp','fn','tp'))
            assert sum(cm.values()) == dataset['test_rows']
            assert abs(model['recall'] - tp/(tp+fn)) < 1e-12
            assert abs(model['false_alarm_rate'] - fp/(fp+tn)) < 1e-12
            assert abs(model['f1'] - 2*tp/(2*tp+fp+fn)) < 1e-12
