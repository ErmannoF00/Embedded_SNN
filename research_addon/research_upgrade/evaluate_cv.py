"""Five-fold subject-held-out logistic and MLP evaluation."""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from .feature_protocol import compute_features,contract
from .data_protocol import validate_dataset,assert_split_integrity,Split
from .evaluate_baselines import score


def confidence_interval(y,pred,groups,seed=41,reps=1000,alpha=.05):
    """Simple cluster bootstrap over unique subject IDs, not windows.

    Not BCa; intended as descriptive uncertainty, not a definitive
    population-level hypothesis test under inter-subject dependence.
    """
    rng=np.random.default_rng(seed)
    ids=np.unique(groups)
    if len(ids)<2:raise ValueError('Requires at least two subjects for CI')
    acc=[];bal=[];f1=[]
    for _ in range(reps):
        draws=rng.choice(ids,size=len(ids),replace=True)
        idx=np.concatenate([np.flatnonzero(groups==gi) for gi in draws])
        # Some bootstrap draws may contain one class only. Balanced accuracy
        # then is ill-defined, so omit those draws for balanced CI.
        measured=score(y[idx],pred[idx]);acc.append(measured['accuracy'])
        if len(np.unique(y[idx]))==2:bal.append(measured['balanced_accuracy'])
        f1.append(measured['macro_f1'])
    def q(v):
        return [float(np.quantile(v,alpha/2)),float(np.quantile(v,1-alpha/2))] if v else None
    return {'accuracy_ci95':q(acc),'balanced_accuracy_ci95':q(bal),
            'macro_f1_ci95':q(f1),'bootstrap_samples':reps,
            'n_subjects':int(len(ids))}


def evaluate(data,output,folds=5,seed=41,reps=1000,feature_mode="amplitude32"):
    with np.load(data,allow_pickle=False) as f:
        x=f['windows'];y=f['labels'];g=f['groups'];fs=float(f['sample_rate_hz'])
    validate_dataset(x,y,g,fs)
    n_groups=len(np.unique(g))
    if n_groups<max(folds,6):raise ValueError('Insufficient subjects for requested folds')
    features=compute_features(x,feature_mode)
    outer=GroupKFold(n_splits=folds,shuffle=True,random_state=seed)
    out={'task':'user-provided labels; verify provenance before claims',
         'folds':folds,'seed':seed,'groups':int(n_groups),
         'models':{},'fold_results':[],'feature_contract':contract(feature_mode)}
    pred={name:np.full(len(x),-1,dtype=np.int64) for name in ('logistic','mlp_48')}
    test_seen=np.zeros(len(x),dtype=bool)
    for fold,(pool,test) in enumerate(outer.split(features,y,g)):
        # Take validation group-disjoint from train pool; do not fit scaler
        # on either validation or test. Hyperparams remain fixed for fairness.
        found=False
        for attempt in range(100):
            tr_rel,va_rel=next(GroupShuffleSplit(n_splits=1,test_size=.2,
                random_state=seed+fold+attempt).split(features[pool],y[pool],g[pool]))
            train=pool[tr_rel];val=pool[va_rel]
            if all(len(np.unique(y[ix]))==2 for ix in (train,val,test)):
                found=True;break
        if not found:raise ValueError(f'Fold {fold}: cannot fit balanced group splits')
        assert_split_integrity(x,g,Split(train,val,test))
        if test_seen[test].any():raise ValueError('Test windows repeated in folds')
        test_seen[test]=True
        scaler=StandardScaler().fit(features[train])
        ft=scaler.transform(features[train]);fv=scaler.transform(features[val]);fe=scaler.transform(features[test])
        models=(('logistic',LogisticRegression(class_weight='balanced',max_iter=2000,random_state=seed)),
                ('mlp_48',MLPClassifier(hidden_layer_sizes=(48,),max_iter=400,
                                     alpha=1e-4,random_state=seed)))
        fold_result={'fold':fold,'groups_train':[str(v) for v in np.unique(g[train])],
                     'groups_val':[str(v) for v in np.unique(g[val])],
                     'groups_test':[str(v) for v in np.unique(g[test])],
                     'count_test':len(test),'models':{}}
        for name,clf in models:
            clf.fit(ft,y[train]);pv=clf.predict(fv);pe=clf.predict(fe)
            pred[name][test]=pe
            fold_result['models'][name]={'val':score(y[val],pv),'test':score(y[test],pe)}
        out['fold_results'].append(fold_result)
    if not test_seen.all():raise ValueError('Not all EEG examples appeared in held-out folds')
    for name,p in pred.items():
        if (p<0).any():raise ValueError('Unfilled prediction')
        per_subject={str(k):score(y[g==k],p[g==k]) for k in np.unique(g)}
        out['models'][name]={'overall_oof':score(y,p),
                             'subject_bootstrap_ci':confidence_interval(y,p,g,seed,reps),
                             'per_subject':per_subject}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,indent=2));print(json.dumps({'out':str(output),
         'models':{k:v['overall_oof'] for k,v in out['models'].items()}},indent=2))
    return out


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--data',required=True)
    parser.add_argument('--output',default='results/subject_folds.json')
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--features',choices=('amplitude32','spectral32'),default='amplitude32')
    a=parser.parse_args();evaluate(a.data,a.output,a.folds,reps=a.bootstrap,feature_mode=a.features)
