"""Subject-held-out supervised ALIF cross-validation."""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .data_protocol import Split, validate_dataset, assert_split_integrity
from .evaluate_baselines import score
from .evaluate_cv import confidence_interval
from .feature_protocol import compute_features,contract


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('PyTorch is needed for SNN training; install requirements-train.txt') from exc
    return torch


def fold_indices(y, groups, folds=5, seed=41):
    """Same GroupKFold and inner group split as evaluate_cv.py, with guards."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    if folds < 2 or np.unique(groups).size < max(folds, 6):
        raise ValueError('At least max(folds,6) subjects needed')
    outer = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (pool, test) in enumerate(outer.split(np.zeros((len(y), 1)), y, groups)):
        for attempt in range(100):
            tr_rel, va_rel = next(GroupShuffleSplit(
                n_splits=1, test_size=.2, random_state=seed+fold+attempt
            ).split(np.zeros((len(pool), 1)), y[pool], groups[pool]))
            tr, val = pool[tr_rel], pool[va_rel]
            if all(np.unique(y[idx]).size == 2 for idx in (tr, val, test)):
                yield fold, Split(tr, val, test)
                break
        else:
            raise ValueError(f'Fold {fold} cannot produce two-class, group-disjoint partitions')


def create_model(steps=20):
    torch = require_torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class Spike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, membrane):
            ctx.save_for_backward(membrane)
            return (membrane >= 0).to(membrane.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (membrane,) = ctx.saved_tensors
            # SuperSpike-style derivative; forward remains an exact hard threshold.
            return grad_output / (1.0 + membrane.abs()).square()

    class ALIF(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(32, 48)
            self.readout = nn.Linear(48, 2)
            self.steps = steps

        def forward(self, features):
            if features.ndim != 2 or features.shape[1] != 32:
                raise ValueError('Normalized features must be (batch,32)')
            current = self.input(features)  # frozen across recurrent SNN steps
            v = torch.zeros_like(current)
            a = torch.zeros_like(current)
            o = features.new_zeros((features.shape[0], 2))
            total_spikes = features.new_zeros(())
            for _ in range(self.steps):
                candidate = .85*v + current - .20*a
                spikes = Spike.apply(candidate - 1.)
                v = candidate - spikes
                a = .90*a + spikes
                o = .90*o + self.readout(spikes)
                total_spikes = total_spikes + spikes.sum()
            return o, total_spikes / (features.shape[0]*48*self.steps)

    return ALIF()


def _predict(model, z, batch_size=256):
    torch = require_torch()
    model.eval()
    out=[]
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            x=torch.from_numpy(np.ascontiguousarray(z[start:start+batch_size], dtype=np.float32))
            logits,_=model(x)
            out.append(logits.numpy())
    return np.concatenate(out) if out else np.zeros((0,2), dtype=np.float32)


def _weights(model):
    return {"w1":model.input.weight.detach().numpy().copy(),
            "b1":model.input.bias.detach().numpy().copy(),
            "w2":model.readout.weight.detach().numpy().copy(),
            "b2":model.readout.bias.detach().numpy().copy()}


def _hash_weights(weights):
    h=hashlib.sha256()
    for k in ('w1','b1','w2','b2'):
        h.update(np.ascontiguousarray(weights[k],dtype=np.float32).tobytes())
    return h.hexdigest()


def evaluate(data, output='results/snn_cv.json', artifacts='artifacts/snn_cv',
             folds=5, epochs=30, batch_size=32, seed=41, lr=.003,
             weight_decay=1e-4, patience=8, bootstrap=1000, feature_mode="amplitude32"):
    torch=require_torch()
    if epochs < 1 or patience < 1 or batch_size < 1 or lr <= 0:
        raise ValueError('Invalid training hyperparameters')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    # Deterministic CPU kernels where supported; no GPU, no mixed precision.
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    with np.load(data,allow_pickle=False) as f:
        x=f['windows'];y=f['labels'].astype(np.int64);g=f['groups']
        fs=float(f['sample_rate_hz'])
    validate_dataset(x,y,g,fs)
    feats=compute_features(x,feature_mode)
    results={'model':'supervised_alif_32_48_2_20_steps', 'seed':seed,
             'folds':folds,'epochs_budget':epochs,'batch_size':batch_size,
             'learning_rate':lr,'weight_decay':weight_decay,
             'selection':'best validation macro F1 (tie break by val loss)',
             'fold_results':[], 'source_data':str(data),
             'feature_contract':contract(feature_mode)}
    predictions=np.full(len(y),-1,dtype=np.int64)
    seen=np.zeros(len(y),dtype=bool)
    base=Path(artifacts)
    base.mkdir(parents=True,exist_ok=True)
    import torch.nn.functional as F
    for fold,split in fold_indices(y,g,folds,seed):
        assert_split_integrity(x,g,split)
        train,val,test=split.train,split.val,split.test
        if seen[test].any(): raise RuntimeError('Overlapping outer fold test subjects')
        seen[test]=True
        scaler=StandardScaler().fit(feats[train])
        # Export scalers in the inference precision.
        mean=scaler.mean_.astype(np.float32)
        scale=np.maximum(scaler.scale_.astype(np.float32),1e-6)
        z=((feats-mean)/scale).astype(np.float32)
        model=create_model()
        optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
        gtor=torch.Generator().manual_seed(seed+fold)
        train_x=torch.from_numpy(z[train]);train_y=torch.from_numpy(y[train])
        best=None;best_rank=None;best_epoch=0;stale=0
        history=[]
        for epoch in range(1,epochs+1):
            model.train()
            perm=torch.randperm(len(train),generator=gtor)
            for idx in perm.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                logits,_=model(train_x[idx])
                loss=F.cross_entropy(logits,train_y[idx])
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step()
            pv=_predict(model,z[val]);val_y=torch.from_numpy(y[val])
            val_loss=float(F.cross_entropy(torch.from_numpy(pv),val_y).item())
            val_f1=score(y[val],pv.argmax(axis=1))['macro_f1']
            history.append({'epoch':epoch,'validation_macro_f1':float(val_f1),
                            'validation_cross_entropy':val_loss})
            rank=(round(float(val_f1),10),-val_loss)
            if best_rank is None or rank>best_rank:
                best_rank=rank;best=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
            else:
                stale+=1
            if stale>=patience:break
        model.load_state_dict(best)
        weights=_weights(model)
        wsha=_hash_weights(weights)
        val_scores=_predict(model,z[val]);test_scores=_predict(model,z[test])
        ypred=np.argmax(test_scores,axis=1).astype(np.int64)
        predictions[test]=ypred
        fdir=base/f'fold_{fold:02d}'
        fdir.mkdir(parents=True,exist_ok=True)
        (fdir/'feature_contract.json').write_text(json.dumps(contract(feature_mode),indent=2))
        np.savez_compressed(fdir/'weights.npz',**weights)
        np.savez_compressed(fdir/'normalizer.npz',mean=mean,scale=scale,
                            selected_epoch=np.array(best_epoch))
        np.savez_compressed(fdir/'validation.npz',z=z[val],labels=y[val],groups=g[val])
        np.savez_compressed(fdir/'test.npz',z=z[test],labels=y[test],groups=g[test])
        fold_report={'fold':fold,'n_train':len(train),'n_val':len(val),'n_test':len(test),
                     'groups_train':[str(v) for v in np.unique(g[train])],
                     'groups_val':[str(v) for v in np.unique(g[val])],
                     'groups_test':[str(v) for v in np.unique(g[test])],
                     'best_epoch':best_epoch,'sha256_weights':wsha,
                     'validation':score(y[val],val_scores.argmax(axis=1)),
                     'test':score(y[test],ypred),'history':history,
                     'artifacts_dir':str(fdir)}
        results['fold_results'].append(fold_report)
    if not seen.all() or np.any(predictions<0):
        raise RuntimeError('Incomplete out-of-fold predictions')
    results['overall_oof']=score(y,predictions)
    results['per_subject']={str(k):score(y[g==k],predictions[g==k]) for k in np.unique(g)}
    results['subject_bootstrap_ci']=confidence_interval(y,predictions,g,seed,bootstrap)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2))
    print(json.dumps({'output':str(output),'overall_oof':results['overall_oof']},indent=2))
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--output',default='results/snn_cv.json')
    p.add_argument('--artifacts',default='artifacts/snn_cv')
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--patience',type=int,default=8)
    p.add_argument('--seed',type=int,default=41)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--features',choices=('amplitude32','spectral32'),default='amplitude32')
    a=p.parse_args()
    evaluate(a.data,a.output,a.artifacts,a.folds,a.epochs,a.batch_size,
             a.seed,patience=a.patience,bootstrap=a.bootstrap,feature_mode=a.features)

if __name__=='__main__':main()
