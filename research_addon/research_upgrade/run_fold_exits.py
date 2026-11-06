"""Evaluate validated adaptive inference using every held-out SNN fold.

Consumes the exact model/scaler and validation/test exports from train_snn_cv.
Does not retrain or tune against outer-test labels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .exit_study import study


def evaluate_all(artifacts='artifacts/snn_cv', output='results/snn_exit_cv.json',
                 tolerated_validation_disagreement=0.0):
    root=Path(artifacts)
    dirs=sorted(d for d in root.glob('fold_*') if d.is_dir())
    if not dirs:raise ValueError('No fold directories containing exported weights and groups')
    fold_reports=[]
    for fold_dir in dirs:
        report=study(fold_dir/'validation.npz',fold_dir/'test.npz',
                     fold_dir/'weights.npz',fold_dir/'exit_study.json',
                     tolerated_validation_disagreement=tolerated_validation_disagreement)
        fold_reports.append({'fold':fold_dir.name,'test':report['test'],
                             'selected_margin_policy':report['selected_margin_policy']})
    modes=('full','certified','calibrated_margin')
    aggregates={}
    for mode in modes:
        rows=[r['test'][mode] for r in fold_reports if mode in r['test']]
        if not rows:continue
        n=sum(r['n'] for r in rows)
        aggregates[mode]={'n':n,'weighted_accuracy':sum(r['n']*r['accuracy'] for r in rows)/n,
                          'weighted_mean_steps':sum(r['n']*r['mean_steps'] for r in rows)/n,
                          'weighted_early_exit_fraction':sum(r['n']*r['early_exit_fraction'] for r in rows)/n,
                          'baseline_disagreements':sum(r['baseline_disagreement'] for r in rows),
                          'certified_exits':sum(r['certified_exits'] for r in rows)}
    result={'folds':fold_reports,'aggregate':aggregates,
            'warning':'Numbers measure inference-step savings, not CPU cycles or energy. '
                      'A certificate concerns final-T classification in exact arithmetic.'}
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2))
    print(json.dumps({'aggregate':aggregates,'output':str(dest)},indent=2))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--artifacts',default='artifacts/snn_cv')
    ap.add_argument('--output',default='results/snn_exit_cv.json')
    ap.add_argument('--allowed-val-disagreement',type=float,default=0.0)
    args=ap.parse_args()
    evaluate_all(args.artifacts,args.output,args.allowed_val_disagreement)
