"""PhysioNet motor-imagery left/right fist EDF-to-window adapter."""
import argparse
import re
from pathlib import Path
import numpy as np

CHANNELS = ('FC3','C3','CP3','FC4','C4','CP4','Cz','Pz')
RUNS = (4,8,12)
OFFSETS_S = (0.5,1.5,2.5)
WINDOW_S = 1.0


def extract_from_raw(raw, subject, channel_names=CHANNELS, *, run=None):
    """Convert one standardized/referenced MNE Raw to (windows,labels,groups).

    Excludes trials with invalid duration and windows that exceed recording.
    No continuous-data filter is applied. MNE EEG signals are in volts;
    conversion to microvolts is intentional and must match acquisition.
    """
    if run is not None and run not in RUNS:
        raise ValueError(f'Expected unilateral motor imagery run in {RUNS}, got {run}')
    fs=float(raw.info['sfreq'])
    if not np.isclose(fs,250.0,atol=1e-8):
        raise ValueError('Call resample(250) before window extraction')
    available={n.casefold():n for n in raw.ch_names}
    missing=[n for n in channel_names if n.casefold() not in available]
    if missing:
        raise ValueError('Missing fixed channel(s): '+str(missing))
    picks=[available[n.casefold()] for n in channel_names]
    windows=[];labels=[];groups=[];trials=[]
    for trial,a in enumerate(raw.annotations):
        annotation=a['description'].strip()
        if annotation not in ('T1','T2'):
            continue
        onset=float(a['onset']);duration=float(a['duration'])
        if duration < (max(OFFSETS_S)+WINDOW_S)-1e-4:
            continue
        for offset in OFFSETS_S:
            t0=onset+offset
            start=int(round(t0*fs));end=start+250
            if end>raw.n_times or end>int(round((onset+duration)*fs))+1:
                continue
            array=raw.get_data(picks=picks,start=start,stop=end)
            if array.shape!=(len(picks),250):
                continue
            window=(array.T*1e6).astype(np.float32)
            if not np.isfinite(window).all():
                continue
            windows.append(window);labels.append(0 if annotation=='T1' else 1)
            groups.append(str(subject))
            trial_key = f'{subject}:R{run:02d}:A{trial:03d}' if run is not None else f'{subject}:UNKNOWN:A{trial:03d}'
            trials.append(trial_key)
    if not windows:
        return (np.empty((0,250,len(picks)),dtype=np.float32),
                np.empty(0,dtype=np.int64),np.empty(0,dtype='U32'),np.empty(0,dtype='U64'))
    return np.stack(windows),np.asarray(labels,dtype=np.int64),\
           np.asarray(groups,dtype='U32'),np.asarray(trials,dtype='U64')


def build(subjects,output,download_dir=None):
    import mne
    from mne.datasets import eegbci
    if not subjects or min(subjects)<1 or max(subjects)>109:
        raise ValueError('subjects must be in [1,109]')
    xs=[];ys=[];gs=[];ts=[]
    for subject in subjects:
        files=eegbci.load_data(subject,list(RUNS),path=download_dir,update_path=False)
        for filename in files:
            match=re.search(r'S(\d{3})R(\d{2})\.edf$',Path(filename).name,re.IGNORECASE)
            if not match or int(match.group(1))!=subject:
                raise ValueError(f'Unexpected EDF filename for subject {subject}: {filename}')
            run=int(match.group(2))
            if run not in RUNS:
                raise ValueError(f'Unexpected task run {run}: {filename}')
            raw=mne.io.read_raw_edf(filename,preload=True,verbose='ERROR')
            eegbci.standardize(raw)
            raw.resample(250,method='polyphase',verbose='ERROR')
            x,y,g,t=extract_from_raw(raw,subject,run=run)
            if len(x):
                xs.append(x);ys.append(y);gs.append(g);ts.append(t)
            raw.close()
    if not xs:
        raise ValueError('No annotated motor-imagery trials extracted')
    x=np.concatenate(xs);y=np.concatenate(ys);g=np.concatenate(gs);t=np.concatenate(ts)
    for ident in np.unique(t):
        idx=(t==ident)
        if idx.sum()>len(OFFSETS_S) or len(np.unique(y[idx]))!=1:
            raise ValueError(f'Non-unique or inconsistent trial identity: {ident}')
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,windows=x,labels=y,groups=g,
        sample_rate_hz=np.array(250.0),source=np.array('PhysioNet EEGMMIDB v1.0.0'),
        task=np.array('imagery of unilateral left vs right fist; runs 4/8/12'),
        channel_names=np.asarray(CHANNELS,dtype='U8'),
        trial_ids=t,adapter_revision=np.array(3),original_sample_rate_hz=np.array(160.0),
        amplitudes_unit=np.array('microvolt'),subjects=np.asarray(subjects,dtype=np.int32))
    print(f'Saved {len(x)} EEG windows from {len(np.unique(t))} unique trials and {len(np.unique(g))} subjects to {dest}')
    return dest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--subjects',nargs='+',type=int,required=True)
    p.add_argument('--output',default='data/physionet_mi_250hz_8ch.npz')
    p.add_argument('--download-dir',default=None)
    a=p.parse_args();build(a.subjects,a.output,a.download_dir)
