"""Reproducible, resource-bounded benchmark and portable decision-tree export.

Install requirements-training.txt, then run from the repository root. Original
Colab outputs are not reproduced verbatim: this script samples each dataset and
keeps SMOTE inside CV folds to avoid preprocessing leakage.
"""
import argparse
import json
import platform
from pathlib import Path

import imblearn
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

FEATURES = ['PROTOCOL', 'TCP_FLAGS', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']
DATASETS = {'NF-UNSW-NB15': 'NF-UNSW-NB15.parquet', 'NF-BoT-IoT': 'NF-BoT-IoT.parquet', 'NF-ToN-IoT': 'NF-ToN-IoT.parquet'}
ROOT = Path(__file__).resolve().parents[1]


def export_tree(pipeline, metadata):
    scaler, model = pipeline.named_steps['scale'], pipeline.named_steps['model']
    tree = model.tree_
    return {'format_version': 1, 'features': FEATURES, 'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(), 'children_left': tree.children_left.tolist(),
            'children_right': tree.children_right.tolist(), 'feature': tree.feature.tolist(),
            'threshold': tree.threshold.tolist(), 'values': tree.value[:, 0, :].tolist(),
            'classes': model.classes_.tolist(), 'metadata': metadata}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--sample-size', type=int, default=15000)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--trees', type=int, default=60)
    args = parser.parse_args()
    if args.sample_size < 100 or args.folds < 2 or args.trees < 1: parser.error('sample-size >= 100, folds >= 2 and trees >= 1 required')
    results = {'method': {'sample_size_per_dataset': args.sample_size, 'split': 'stratified 80/20, random_state=42',
      'cv': f'{args.folds}-fold stratified macro F1 on training data; scaler and SMOTE fit inside each fold',
      'trees': args.trees, 'random_state': 42, 'tuning': 'none', 'training_data': 'public Kaggle parquet datasets'},
      'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__, 'imbalanced_learn': imblearn.__version__}, 'datasets': {}}
    exported = None
    for name, filename in DATASETS.items():
        frame = pd.read_parquet(args.data_dir / filename, columns=FEATURES + ['Label'])
        total = len(frame)
        if frame[FEATURES + ['Label']].isna().any().any(): raise ValueError(f'{name}: missing values; inspect before training')
        if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all(): raise ValueError(f'{name}: nonfinite features')
        if set(frame.Label.unique()) != {0, 1}: raise ValueError(f'{name}: expected binary labels')
        if len(frame) > args.sample_size:
            frame, _ = train_test_split(frame, train_size=args.sample_size, stratify=frame.Label, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(frame[FEATURES], frame.Label.astype(int), test_size=.2, stratify=frame.Label, random_state=42)
        group = {'source_rows': total, 'sample_rows': len(frame), 'train_rows': len(X_train), 'test_rows': len(X_test),
                 'sample_class_counts': {str(int(k)): int(v) for k,v in frame.Label.value_counts().items()},
                 'test_class_counts': {str(int(k)): int(v) for k,v in y_test.value_counts().items()}, 'models': {}}
        candidates = {'Random Forest': RandomForestClassifier(n_estimators=args.trees,max_depth=15,random_state=42,n_jobs=2),
          'Decision Tree': DecisionTreeClassifier(max_depth=15,random_state=42),
          'XGBoost': XGBClassifier(n_estimators=args.trees,max_depth=6,learning_rate=.1,random_state=42,n_jobs=2,eval_metric='logloss')}
        for model_name, model in candidates.items():
            print(f'{name}: {model_name}', flush=True)
            pipe=Pipeline([('scale',StandardScaler()),('smote',SMOTE(random_state=42)),('model',model)])
            cv=cross_val_score(pipe,X_train,y_train,cv=args.folds,scoring='f1_macro',n_jobs=1)
            pipe.fit(X_train,y_train)
            pred=pipe.predict(X_test); prob=pipe.predict_proba(X_test)[:,1]
            tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
            group['models'][model_name]={'accuracy':float(accuracy_score(y_test,pred)), 'f1':float(f1_score(y_test,pred,zero_division=0)),
              'precision':float(precision_score(y_test,pred,zero_division=0)), 'recall':float(recall_score(y_test,pred,zero_division=0)),
              'roc_auc':float(roc_auc_score(y_test,prob)), 'false_alarm_rate':float(fp/(fp+tn)), 'cv_macro_f1_mean':float(cv.mean()),
              'cv_macro_f1_std':float(cv.std()), 'confusion_matrix':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}}
            if name == 'NF-UNSW-NB15' and model_name == 'Decision Tree':
                exported=export_tree(pipe, {'dataset':name,'algorithm':model_name,'training_sample_rows':len(frame), 'seed':42,
                                           'note':'Experimental research model, not a production security control'})
                # Verify exported arithmetic exactly against sklearn on the held-out data.
                from backend.services.predictor import predict_tree
                for row, expected in zip(X_test.to_numpy()[:300], pred[:300]):
                    if predict_tree(exported, dict(zip(FEATURES, map(float,row))))['label'] != int(expected):
                        raise AssertionError('portable tree differs from scikit-learn')
        results['datasets'][name]=group
        print(f'{name}: done',flush=True)
    (ROOT/'ml'/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (ROOT/'backend'/'models'/'decision_tree.json').write_text(json.dumps(exported,separators=(',',':'))+'\n')
    print('Results and portable model saved',flush=True)

if __name__ == '__main__': main()
