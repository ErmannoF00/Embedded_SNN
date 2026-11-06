import io, subprocess, shutil, os
from pathlib import Path
import numpy as np
import pytest
from research_upgrade.certified_exit import Parameters,infer
from research_upgrade.export_cpp import export


def test_host_cpp_parity(tmp_path):
    compiler=shutil.which('g++')
    if not compiler: pytest.skip('no C++ compiler')
    rng=np.random.default_rng(41)
    p=Parameters(rng.normal(0,.1,(48,32)).astype('f4'),
                 rng.normal(0,.1,48).astype('f4'),
                 rng.normal(0,.1,(2,48)).astype('f4'),
                 rng.normal(0,.1,2).astype('f4'))
    weights=tmp_path/'weights.npz'
    np.savez_compressed(weights,w1=p.w1,b1=p.b1,w2=p.w2,b2=p.b2)
    export(weights,tmp_path/'weights.hpp')
    src=Path(__file__).resolve().parents[1]/'cpp'/'host_cli.cpp'
    binary=tmp_path/'host'
    subprocess.run([compiler,'-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra',
                    '-I'+str(src.parent),'-I'+str(tmp_path),str(src),'-o',str(binary)],check=True)
    x=rng.normal(0,1,(40,32)).astype('f4')
    payload='\n'.join(' '.join(f'{float(v):.9g}' for v in row) for row in x)+'\n'
    for mode in ('full','certified'):
        cp=subprocess.run([str(binary),mode],input=payload,text=True,capture_output=True,check=True)
        lines=[row.strip().split(',') for row in cp.stdout.strip().splitlines()]
        assert len(lines)==len(x)
        for i,row in enumerate(lines):
            gold=infer(x[i],p,mode=mode)
            np.testing.assert_allclose(np.asarray(row[:2],dtype=float),gold.scores,atol=1e-4,rtol=1e-5)
            assert int(row[2])==gold.pred
            assert int(row[3])==gold.steps_used
