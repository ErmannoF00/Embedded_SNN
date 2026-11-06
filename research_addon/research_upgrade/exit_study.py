"""Validation-only threshold selection, untouched test for adaptive ALIF policies.

Requires NORMALIZED features and weights saved from exact same checkpoint.
No synthetic fixture is presented as real EEG results. Baseline agreement
is not correctness against actual experimental labels.
"""
import argparse,json
from pathlib import Path
import numpy as np
from .certified_exit import Parameters, Config, evaluate_exit


def _check_data(z,labels,groups):
    z=np.asarray(z,dtype=np.float32); y=np.asarray(labels);g=np.asarray(groups)
    if z.ndim!=2 or z.shape[1]!=32 or len(z)!=len(y) or len(z)!=len(g):
        raise ValueError('Invalid normalized features/labels/groups')
    if not np.isfinite(z).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Bad features/labels')
    return z,y,g


def study(val_npz,test_npz,weights_npz,output,thresholds=(0.25,0.5,1.,2.,4.,8.),
          tolerated_validation_disagreement=0.0,min_steps=2):
    with np.load(val_npz,allow_pickle=False) as a:
        zv,yv,gv=_check_data(a['z'],a['labels'],a['groups'])
    with np.load(test_npz,allow_pickle=False) as a:
        zt,yt,gt=_check_data(a['z'],a['labels'],a['groups'])
    if np.intersect1d(gv,gt).size:
        raise ValueError('Validation and test GROUPS overlap')
    p=Parameters.from_npz(str(weights_npz));cfg=Config()
    validations=[]
    for th in thresholds:
        r=evaluate_exit(zv,p,cfg,mode='margin',margin_threshold=float(th),min_steps=min_steps)
        validations.append({'threshold':float(th),'disagreement_fraction':r['disagreements']/r['n'],
                            'mean_steps':float(r['steps'].mean()),'p95_steps':float(np.percentile(r['steps'],95))})
    eligible=[x for x in validations if x['disagreement_fraction']<=tolerated_validation_disagreement]
    best=min(eligible,key=lambda x:(x['mean_steps'],-x['threshold'])) if eligible else None
    methods={'full':{},'certified':{}}
    if best is not None:
        methods['calibrated_margin']={'threshold':best['threshold']}
    results={'warnings':['Certificate proves baseline class only in exact arithmetic; not true EEG correctness.',
                          'Margin threshold tuned only on held-out validation; zero validation errors do not guarantee test reliability.'],
             'validation':validations,'selected_margin_policy':best,'test':{}}
    for name,options in methods.items():
        mode='margin' if name=='calibrated_margin' else name
        r=evaluate_exit(zt,p,cfg,mode=mode,min_steps=min_steps,
                        **({'margin_threshold':options['threshold']} if options else {}))
        pred=r['policy_predictions']
        results['test'][name]={'n':r['n'],'accuracy':float((pred==yt).mean()),
            'baseline_disagreement':int(r['disagreements']),
            'mean_steps':float(r['steps'].mean()),
            'p95_steps':float(np.percentile(r['steps'],95)),
            'early_exit_fraction':float((r['steps']<cfg.steps).mean()),
            'certified_exits':int(r['certified_exits']),
            'mean_hidden_spikes':float(r['spikes'].mean())}
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2));return results


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--val',required=True,help='npz: z, labels, groups')
    p.add_argument('--test',required=True,help='npz: z, labels, groups')
    p.add_argument('--weights',required=True,help='npz: w1,b1,w2,b2 from SAME checkpoint')
    p.add_argument('--output',default='results/exit_study.json')
    p.add_argument('--allowed-val-disagreement',type=float,default=0.)
    a=p.parse_args();study(a.val,a.test,a.weights,a.output,
                             tolerated_validation_disagreement=a.allowed_val_disagreement)
