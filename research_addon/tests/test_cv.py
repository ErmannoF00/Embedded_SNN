from research_upgrade.evaluate_cv import evaluate
import numpy as np


def test_group_cv_smoke(tmp_path):
    rng=np.random.default_rng(888)
    x=rng.standard_normal((100,250,8)).astype('f4')
    labels=np.tile([0,1],50)
    x[labels==1,:,:]+=0.1
    groups=np.repeat(np.arange(10),10)
    file=tmp_path/'fixture.npz'
    np.savez_compressed(file,windows=x,labels=labels,groups=groups,sample_rate_hz=250.)
    result=evaluate(file,tmp_path/'cv.json',folds=5,seed=41,reps=25)
    assert len(result['fold_results'])==5
    assert all(len(set(v['groups_train'])&set(v['groups_test']))==0 for v in result['fold_results'])
    assert result['models']['logistic']['subject_bootstrap_ci']['n_subjects']==10
