"""Compare a trained fold against the C++17 host inference implementation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from .certified_exit import Parameters, infer
from .export_cpp import export


def verify(fold_dir, output='results/trained_cpp_parity.json', limit=128,
           atol=5e-4, rtol=1e-5, cxx='g++'):
    root=Path(fold_dir)
    weights=root/'weights.npz'
    test=root/'test.npz'
    if not weights.exists() or not test.exists():
        raise FileNotFoundError('Expected weights.npz and test.npz in a trained fold directory')
    if shutil.which(cxx) is None:
        raise RuntimeError(f'C++ compiler unavailable: {cxx}')
    if limit<1 or atol<0 or rtol<0:
        raise ValueError('limit>=1 and nonnegative tolerances required')
    with np.load(test,allow_pickle=False) as f:
        z=np.asarray(f['z'],dtype=np.float32)
        y=np.asarray(f['labels'],dtype=np.int64)
    if z.ndim!=2 or z.shape[1]!=32 or len(z)!=len(y) or not np.isfinite(z).all():
        raise ValueError('Malformed saved fold test features')
    z=z[:limit]
    params=Parameters.from_npz(str(weights))
    addon=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='embedded_snn_cpp_parity_') as tdir:
        tmp=Path(tdir)
        export(weights,tmp/'weights.hpp')
        binary=tmp/'research_snn_host'
        build=subprocess.run([cxx,'-std=c++17','-O2','-ffp-contract=off',
                              '-Wall','-Wextra','-Wpedantic',
                              '-I'+str(addon/'cpp'),'-I'+str(tmp),
                              str(addon/'cpp'/'host_cli.cpp'),'-o',str(binary)],
                             text=True,capture_output=True)
        if build.returncode:
            raise RuntimeError(f'C++ build failed:\n{build.stderr}')
        payload='\n'.join(' '.join(format(float(v),'.9g') for v in row) for row in z)+'\n'
        run=subprocess.run([str(binary),'full'],input=payload,text=True,
                           capture_output=True)
        if run.returncode:
            raise RuntimeError(f'C++ inference failed ({run.returncode}): {run.stderr}')
        rows=[line.strip().split(',') for line in run.stdout.splitlines() if line.strip()]
    if len(rows)!=len(z) or any(len(row)!=5 for row in rows):
        raise RuntimeError('C++ CLI emitted unexpected number or format of rows')
    cpp_scores=np.asarray([[float(r[0]),float(r[1])] for r in rows],dtype=np.float64)
    cpp_labels=np.asarray([int(r[2]) for r in rows],dtype=np.int64)
    cpp_steps=np.asarray([int(r[3]) for r in rows],dtype=np.int64)
    py=[infer(x,params,mode='full') for x in z]
    py_scores=np.asarray([result.scores for result in py],dtype=np.float64)
    py_labels=np.asarray([result.pred for result in py],dtype=np.int64)
    diff=np.abs(cpp_scores-py_scores)
    allowed=atol+rtol*np.abs(py_scores)
    mismatches=int(np.count_nonzero(py_labels!=cpp_labels))
    out={'fold':str(root),'n':int(len(z)),
         'feature_contract':(json.loads((root/'feature_contract.json').read_text())
                             if (root/'feature_contract.json').exists() else None),
         'max_absolute_score_error':float(diff.max(initial=0)),
         'max_relative_score_error':float((diff/np.maximum(np.abs(py_scores),1e-12)).max(initial=0)),
         'score_elements_outside_tolerance':int(np.count_nonzero(diff>allowed)),
         'class_mismatches':mismatches,
         'invalid_step_count':int(np.count_nonzero(cpp_steps!=20)),
         'atol':atol,'rtol':rtol,
         'status':'PASS' if mismatches==0 and np.all(diff<=allowed) and np.all(cpp_steps==20) else 'FAIL',
         'notes': 'Temporary C++ build with exact trained weights; normalized-feature parity only, no generated header left behind, not MCU raw-window parity'}
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fold',default='artifacts/snn_cv/fold_00')
    p.add_argument('--output',default='results/trained_cpp_parity.json')
    p.add_argument('--limit',type=int,default=128)
    p.add_argument('--atol',type=float,default=5e-4)
    p.add_argument('--rtol',type=float,default=1e-5)
    args=p.parse_args()
    report=verify(args.fold,args.output,args.limit,args.atol,args.rtol)
    if report['status']!='PASS':
        raise SystemExit('Parity FAILED; do NOT claim the trained model is numerically portable')

if __name__=='__main__':
    main()
