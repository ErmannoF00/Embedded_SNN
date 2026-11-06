#!/usr/bin/env python3
"""Subject-held-out logistic baseline for amplitude and spectral features."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from .feature_protocol import compute_features

BANDS = ((8.0, 12.0), (12.0, 16.0), (16.0, 22.0), (22.0, 30.0))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', default='results/spectral_diagnostic.json')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=41)
    args=ap.parse_args()
    with np.load(args.data, allow_pickle=False) as f:
        required={'windows','labels','groups','sample_rate_hz'}
        if not required.issubset(f.files):
            raise ValueError('Missing NPZ entries: '+str(required-set(f.files)))
        x=f['windows'];y=f['labels'].astype(np.int64);g=f['groups']
        fs=float(f['sample_rate_hz'])
        trials=f['trial_ids'] if 'trial_ids' in f.files else None
    if x.shape[1:] != (250,8) or y.shape!=(len(x),) or g.shape!=(len(x),):
        raise ValueError('Invalid dataset shape')
    if set(np.unique(y)) != {0,1}:
        raise ValueError('Requires binary labels 0/1')
    if len(np.unique(g))<args.folds:
        raise ValueError('Insufficient subjects')
    trial_warning = None
    if trials is not None:
        # Reject trial IDs that collide across recordings.
        counts = np.unique(trials, return_counts=True)[1]
        mixed = any(len(np.unique(y[trials == t])) != 1 for t in np.unique(trials))
        if np.any(counts > 3) or mixed:
            trial_warning = ('trial_ids from v0.2 may collide across recording runs; '
                             'trial-vote metrics suppressed until rebuilt with unique run IDs')
            trials = None
    feats={'original_amplitude':compute_features(x,'amplitude32'),
           'mu_beta_logpower':compute_features(x,'spectral32')}
    splitter=GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    result={'dataset':str(args.data),'n_windows':len(x),
            'n_subjects':len(np.unique(g)),
            'n_trials':len(np.unique(trials)) if trials is not None else None,
            'trial_warning':trial_warning,
            'bands_hz':BANDS, 'methods':{},
            'note':'Subject-held-out exploratory logistic baselines; not MCU-optimized features'}
    splits=list(splitter.split(feats['mu_beta_logpower'], y, g))
    for name,feat in feats.items():
        pred=np.full(len(y),-1,dtype=int)
        fold_stats=[]
        for fold,(tr,te) in enumerate(splits):
            model=make_pipeline(StandardScaler(), LogisticRegression(
                class_weight='balanced',max_iter=2000,random_state=args.seed))
            model.fit(feat[tr],y[tr])
            pred[te]=model.predict(feat[te])
            fold_stats.append({'fold':fold,
                'test_subjects':[str(v) for v in np.unique(g[te])],
                'accuracy':float(accuracy_score(y[te],pred[te])),
                'balanced_accuracy':float(balanced_accuracy_score(y[te],pred[te]))})
        if np.any(pred < 0):
            raise AssertionError('Incomplete out-of-fold predictions')
        subject_scores={str(s):float(balanced_accuracy_score(y[g==s],pred[g==s]))
                        for s in np.unique(g)}
        stats={'accuracy':float(accuracy_score(y,pred)),
               'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
               'macro_f1':float(f1_score(y,pred,average='macro')),
               'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),
               'per_subject_balanced_accuracy':subject_scores,
               'folds':fold_stats}
        if trials is not None:
            trial_y=[];trial_p=[]
            for t in np.unique(trials):
                idx=np.flatnonzero(trials==t)
                trial_y.append(int(y[idx[0]]))
                counts=np.bincount(pred[idx],minlength=2)
                trial_p.append(int(np.argmax(counts)))
            stats['trial_majority_vote_accuracy']=float(accuracy_score(trial_y,trial_p))
            stats['trial_majority_vote_balanced_accuracy']=float(balanced_accuracy_score(trial_y,trial_p))
        result['methods'][name]=stats
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps({'output':str(out), 'n_windows':len(y),
        'n_trials':result['n_trials'],
        'metrics':{k:{a:v[a] for a in ('accuracy','balanced_accuracy','macro_f1','trial_majority_vote_balanced_accuracy') if a in v}
                   for k,v in result['methods'].items()}},indent=2))


if __name__=='__main__':
    main()
