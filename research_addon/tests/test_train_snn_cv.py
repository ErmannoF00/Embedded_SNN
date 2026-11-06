import numpy as np
import pytest
from research_upgrade.train_snn_cv import create_model,fold_indices,evaluate
from research_upgrade.certified_exit import Config,Parameters,infer


def test_training_forward_matches_reference_pretraining():
    torch=pytest.importorskip('torch')
    torch.manual_seed(17)
    model=create_model()
    model.eval()
    x=np.random.default_rng(9).normal(0,0.5,size=(3,32)).astype(np.float32)
    logits,_=model(torch.from_numpy(x))
    p=Parameters(model.input.weight.detach().numpy(),model.input.bias.detach().numpy(),
                 model.readout.weight.detach().numpy(),model.readout.bias.detach().numpy())
    for i in range(3):
        reference=infer(x[i],p,Config(),mode='full')
        np.testing.assert_allclose(logits.detach().numpy()[i],reference.scores,rtol=2e-5,atol=2e-5)


def test_subject_fold_partitions():
    groups=np.repeat(np.arange(10),6)
    labels=np.tile([0,1,0,1,0,1],10)
    covered=np.zeros(len(groups),dtype=int)
    for _,split in fold_indices(labels,groups,folds=5):
        assert not (set(groups[split.train]) & set(groups[split.val]))
        assert not (set(groups[split.train]) & set(groups[split.test]))
        assert not (set(groups[split.val]) & set(groups[split.test]))
        covered[split.test]+=1
    assert np.all(covered==1)


def test_train_short_synthetic_fixture(tmp_path):
    pytest.importorskip('torch')
    rng=np.random.default_rng(5)
    # Each group has both classes. Sinewave examples are SYNTHETIC and
    # explicitly unsuitable as a claim of real EEG classification.
    nsub,nper=6,8
    t=np.arange(250,dtype=np.float32)/250
    xx=[]; yy=[]; gg=[]
    for subject in range(nsub):
        for k in range(nper):
            label=k%2
            signal=(8+4*label)*np.sin(2*np.pi*(8+label*3)*t)
            xx.append((signal[:,None]+rng.normal(0,1,(250,8))).astype(np.float32))
            yy.append(label);gg.append(subject)
    dataset=tmp_path/'synthetic.npz'
    np.savez(dataset,windows=np.stack(xx),labels=np.asarray(yy),
             groups=np.asarray(gg),sample_rate_hz=np.array(250.0))
    output=tmp_path/'metrics.json'
    result=evaluate(dataset,output,tmp_path/'artifacts',folds=3,epochs=2,
                    patience=2,bootstrap=10)
    assert len(result['fold_results'])==3
    assert output.exists()
    for fold in range(3):
        directory=tmp_path/'artifacts'/f'fold_{fold:02d}'
        with np.load(directory/'weights.npz') as f:
            assert f['w1'].shape==(48,32)
        assert (directory/'test.npz').exists()


def test_folds_then_early_exit_study(tmp_path):
    pytest.importorskip('torch')
    from research_upgrade.run_fold_exits import evaluate_all
    rng=np.random.default_rng(13)
    x=rng.normal(0,1,(60,250,8)).astype(np.float32)
    x[:, :, 0] += np.tile([0.,2.],30)[:,None]
    y=np.tile([0,1]*5,6).astype(np.int64)
    g=np.repeat(np.arange(6),10)
    path=tmp_path/'synthetic.npz'
    np.savez(path,windows=x,labels=y,groups=g,sample_rate_hz=np.array(250.0))
    directory=tmp_path/'snn'
    evaluate(path,tmp_path/'snn.json',directory,folds=3,epochs=1,bootstrap=10)
    res=evaluate_all(directory,tmp_path/'exits.json')
    assert res['aggregate']['full']['n']==len(y)
    assert res['aggregate']['certified']['baseline_disagreements']==0
    assert res['aggregate']['full']['weighted_mean_steps']==20
