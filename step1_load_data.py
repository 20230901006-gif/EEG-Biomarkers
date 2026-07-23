"""
STEP 1 - Data Loader
Loads all 14 subjects (7 IDD + 7 TDC) x 2 conditions into a dictionary.
Run this first. Paste the output before moving to Step 2.
"""

import scipy.io as sio
import numpy as np
import os

# ── Config ────────────────────────────────────────────────────
BASE = r'D:\Data science Project\EEG Chrononet\EEGdata\Data\CleanData'

CHANNELS = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
SFREQ    = 128   # Hz
EXPECTED_SHAPE = (14, 15360)  # 14 ch x 120s x 128Hz

# File manifest — (subject_id, group, condition, filepath)
MANIFEST = [
    # IDD subjects
    ('NDS001','IDD','Music', os.path.join(BASE,'CleanData_IDD','Music','NDS001_Music_CD.mat')),
    ('NDS001','IDD','Rest',  os.path.join(BASE,'CleanData_IDD','Rest', 'NDS001_Rest_CD.mat')),
    ('NDS002','IDD','Music', os.path.join(BASE,'CleanData_IDD','Music','NDS002_Music_CD.mat')),
    ('NDS002','IDD','Rest',  os.path.join(BASE,'CleanData_IDD','Rest', 'NDS002_Rest_CD.mat')),
    ('NDS003','IDD','Music', os.path.join(BASE,'CleanData_IDD','Music','NDS003_Music_CD.mat')),
    ('NDS003','IDD','Rest',  os.path.join(BASE,'CleanData_IDD','Rest', 'NDS003_Rest_CD.mat')),
    ('NDS004','IDD','Music', os.path.join(BASE,'CleanData_IDD','Music','NDS004_Music_CD.mat')),
    ('NDS004','IDD','Rest',  os.path.join(BASE,'CleanData_IDD','Rest', 'NDS004_Rest_CD.mat')),
    ('NDS005','IDD','Music', os.path.join(BASE,'CleanData_IDD','Music','NDS005_Music_CD.mat')),
    ('NDS005','IDD','Rest',  os.path.join(BASE,'CleanData_IDD','Rest', 'NDS005_Rest_CD.mat')),
    ('NDS006','IDD','Music', os.path.join(BASE,'CleanData_IDD','Music','NDS006_Music_CD.mat')),
    ('NDS006','IDD','Rest',  os.path.join(BASE,'CleanData_IDD','Rest', 'NDS006_Rest_CD.mat')),
    ('NDS007','IDD','Music', os.path.join(BASE,'CleanData_IDD','Music','NDS007_Music_CD.mat')),
    ('NDS007','IDD','Rest',  os.path.join(BASE,'CleanData_IDD','Rest', 'NDS007_Rest_CD.mat')),
    # TDC subjects
    ('CGS01','TDC','Music', os.path.join(BASE,'CleanData_TDC','Music','CGS01_Music_CD.mat')),
    ('CGS01','TDC','Rest',  os.path.join(BASE,'CleanData_TDC','Rest', 'CGS01_Rest_CD.mat')),
    ('CGS02','TDC','Music', os.path.join(BASE,'CleanData_TDC','Music','CGS02_Music_CD.mat')),
    ('CGS02','TDC','Rest',  os.path.join(BASE,'CleanData_TDC','Rest', 'CGS02_Rest_CD.mat')),
    ('CGS03','TDC','Music', os.path.join(BASE,'CleanData_TDC','Music','CGS03_Music_CD.mat')),
    ('CGS03','TDC','Rest',  os.path.join(BASE,'CleanData_TDC','Rest', 'CGS03_Rest_CD.mat')),
    ('CGS04','TDC','Music', os.path.join(BASE,'CleanData_TDC','Music','CGS04_Music_CD.mat')),
    ('CGS04','TDC','Rest',  os.path.join(BASE,'CleanData_TDC','Rest', 'CGS04_Rest_CD.mat')),
    ('CGS05','TDC','Music', os.path.join(BASE,'CleanData_TDC','Music','CGS05_Music_CD.mat')),
    ('CGS05','TDC','Rest',  os.path.join(BASE,'CleanData_TDC','Rest', 'CGS05_Rest_CD.mat')),
    ('CGS06','TDC','Music', os.path.join(BASE,'CleanData_TDC','Music','CGS06_Music_CD.mat')),
    ('CGS06','TDC','Rest',  os.path.join(BASE,'CleanData_TDC','Rest', 'CGS06_Rest_CD.mat')),
    ('CGS07','TDC','Music', os.path.join(BASE,'CleanData_TDC','Music','CGS07_Music_CD.mat')),
    ('CGS07','TDC','Rest',  os.path.join(BASE,'CleanData_TDC','Rest', 'CGS07_Rest_CD.mat')),
]

# ── Load ──────────────────────────────────────────────────────
print('='*55)
print('  STEP 1 — Loading CleanData')
print('='*55)

data = {}   # data[(subject_id, condition)] = np.array(14, 15360)
errors = []

for subj, group, cond, fpath in MANIFEST:
    key = (subj, cond)
    fname = os.path.basename(fpath)

    if not os.path.exists(fpath):
        print(f'  MISSING  {fname}')
        errors.append(fname)
        continue

    mat  = sio.loadmat(fpath)
    arr  = mat['clean_data'].astype(np.float32)

    if arr.shape != EXPECTED_SHAPE:
        print(f'  SHAPE ERR {fname}  got {arr.shape}')
        errors.append(fname)
        continue

    data[key] = {'eeg': arr, 'group': group, 'subj': subj, 'cond': cond}
    print(f'  OK  {fname:35s}  group={group}  shape={arr.shape}')

# ── Summary ───────────────────────────────────────────────────
print()
print(f'  Loaded : {len(data)} / {len(MANIFEST)} recordings')
print(f'  Errors : {len(errors)}')

if len(data) == len(MANIFEST):
    print('\n  ✅  All files loaded successfully.')
    print(f'  Subjects : 14  |  Conditions : Rest, Music  |  Channels : {len(CHANNELS)}')
    print(f'  Samples  : {EXPECTED_SHAPE[1]} per recording  ({EXPECTED_SHAPE[1]/SFREQ:.0f} s @ {SFREQ} Hz)')
else:
    print('\n  ❌  Some files failed — check paths above.')

print('='*55)
