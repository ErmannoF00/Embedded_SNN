"""Logistic regression and parameter-matched single-hidden-layer MLP baselines.

No SNN is trained here; this script provides a minimum real EEG reference.
For a publication, run multiple subject-wise folds, not only one seed.
"""
import argparse, json, pathlib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from .features import extract_features
from .data_protocol import validate_dataset,split_groups,assert_split_integrity


def score(y_true,y_pred):
    return {'accuracy':float(accuracy_score(y_true,y_pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y_true,y_pred)),
            'macro_f1':float(f1_score(y_true,y_pred,average='macro',zero_division=0)),
            'confusion_matrix':confusion_matrix(y_true,y_pred,labels=[0,1]).tolist()}


def run(data_path,output_path,seed=41):
    with np.load(data_path,allow_pickle=False) as f:
        for key in ('windows','labels','groups','sample_rate_hz'):
            if key not in f:
                raise ValueError('Missing dataset key '+key)
        x=np.asarray(f['windows']);y=np.asarray(f['labels']);g=np.asarray(f['groups']);fs=float(f['sample_rate_hz'])
    validate_dataset(x,y,g,fs)
    split=split_groups(y,g,seed)
    assert_split_integrity(x,g,split)
    F=extract_features(x)
    scaler=StandardScaler().fit(F[split.train])
    Z=scaler.transform(F)
    results={'note':'Single group-disjoint split; do not compare as cross-validated paper result',
             'dataset_path':str(data_path),'seed':seed,
             'n_windows':len(x),'n_groups':len(np.unique(g)),
             'split':{k:{'n':int(len(getattr(split,k))),
                           'groups':[str(t) for t in np.unique(g[getattr(split,k)])]}
                       for k in ('train','val','test')},'models':{}}
    for name,clf in (
        ('logistic',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=seed)),
        ('mlp_48',MLPClassifier(hidden_layer_sizes=(48,),alpha=1e-4,early_stopping=False,
                                max_iter=400,random_state=seed))):
        clf.fit(Z[split.train],y[split.train]); pv=clf.predict(Z[split.val]);pt=clf.predict(Z[split.test])
        subject_scores={str(gi):score(y[subset],pt[np.isin(split.test,subset)])
            for gi in np.unique(g[split.test])
            if (subset:=split.test[g[split.test]==gi]).size>0}
        results['models'][name]={'validation':score(y[split.val],pv),
                                 'test':score(y[split.test],pt),
                                 'test_per_group':subject_scores}
    out=pathlib.Path(output_path);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2))
    print(json.dumps({'output':str(out),'models':{k:v['test'] for k,v in results['models'].items()}},indent=2))
    return results


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--output',default='results/real_eeg_baselines.json')
    ap.add_argument('--seed',type=int,default=41)
    a=ap.parse_args();run(a.data,a.output,a.seed)
