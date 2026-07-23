import scipy.io as sio
import numpy as np

files = [
    r'D:\Data science Project\EEG Chrononet\EEGdata\Data\CleanData\CleanData_IDD\Music\NDS001_Music_CD.mat',
    r'D:\Data science Project\EEG Chrononet\EEGdata\Data\CleanData\CleanData_IDD\Rest\NDS001_Rest_CD.mat',
    r'D:\Data science Project\EEG Chrononet\EEGdata\Data\CleanData\CleanData_IDD\Rest\NDS001_Rest_CD(1).mat',
    r'D:\Data science Project\EEG Chrononet\EEGdata\Data\CleanData\CleanData_TDC\Music\CGS01_Music_CD.mat',
    r'D:\Data science Project\EEG Chrononet\EEGdata\Data\CleanData\CleanData_TDC\Rest\CGS01_Rest_CD.mat',
]

for f in files:
    fname = f.split('\\')[-1]
    print(f'\n{"="*50}')
    print(f'FILE: {fname}')
    print(f'{"="*50}')
    try:
        mat = sio.loadmat(f)
        for k, v in mat.items():
            if not k.startswith('__'):
                if hasattr(v, 'shape'):
                    print(f'  key = {k!r:25s}  shape = {str(v.shape):20s}  dtype = {v.dtype}')
                else:
                    print(f'  key = {k!r:25s}  type  = {type(v).__name__}')
    except Exception as e:
        print(f'  scipy.io.loadmat failed: {e}')
        print('  Trying mat73 ...')
        try:
            import mat73
            mat = mat73.loadmat(f)
            for k, v in mat.items():
                try:
                    arr = np.array(v)
                    print(f'  key = {k!r:25s}  shape = {str(arr.shape):20s}  dtype = {arr.dtype}')
                except Exception:
                    print(f'  key = {k!r:25s}  type  = {type(v).__name__}')
        except Exception as e2:
            print(f'  mat73 also failed: {e2}')

print('\nDone.')
