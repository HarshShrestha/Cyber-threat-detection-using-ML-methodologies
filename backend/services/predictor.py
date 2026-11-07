"""Inference for the exported sklearn decision tree; no pickle or training dependencies."""
import json
import math
import struct
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'decision_tree.json'

@lru_cache(maxsize=1)
def load_model():
    return json.loads(MODEL_PATH.read_text())

def predict_tree(model, features):
    # sklearn's trees coerce input to float32 during prediction.
    values = [struct.unpack('f', struct.pack('f', (float(features[name])-mean)/scale))[0]
              for name,mean,scale in zip(model['features'],model['mean'],model['scale'])]
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Feature values must be finite')
    node = 0
    while model['children_left'][node] != -1:
        node = (model['children_left'][node] if values[model['feature'][node]] <= model['threshold'][node]
                else model['children_right'][node])
    counts = model['values'][node]
    total = sum(counts)
    probs = [float(v/total) for v in counts]
    index = max(range(len(probs)), key=lambda i: probs[i])
    attack_index = model['classes'].index(1)
    return {'label': int(model['classes'][index]), 'attack_probability': probs[attack_index]}
