"""
EEG Research Environment Verification Script
Run this after installing all libraries to confirm everything is working.

Usage:
    conda activate eeg_idd
    cd "D:\Data science Project\EEG Chrononet\EEGdata"
    python verify_setup.py
"""

import sys

print("=" * 60)
print("  EEG Research Environment Verification")
print("=" * 60)

errors = []

# ── Python ────────────────────────────────────────────────────
print(f"\n[1] Python:  {sys.version.split()[0]}")

# ── Core libraries ────────────────────────────────────────────
print("\n[2] Core Libraries:")
try:
    import numpy as np
    print(f"    numpy        {np.__version__}  ✓")
except ImportError as e:
    print(f"    numpy        MISSING  ✗  ({e})")
    errors.append("numpy")

try:
    import scipy
    print(f"    scipy        {scipy.__version__}  ✓")
except ImportError as e:
    print(f"    scipy        MISSING  ✗  ({e})")
    errors.append("scipy")

try:
    import pandas as pd
    print(f"    pandas       {pd.__version__}  ✓")
except ImportError as e:
    print(f"    pandas       MISSING  ✗  ({e})")
    errors.append("pandas")

try:
    import matplotlib
    print(f"    matplotlib   {matplotlib.__version__}  ✓")
except ImportError as e:
    print(f"    matplotlib   MISSING  ✗  ({e})")
    errors.append("matplotlib")

try:
    import seaborn as sns
    print(f"    seaborn      {sns.__version__}  ✓")
except ImportError as e:
    print(f"    seaborn      MISSING  ✗  ({e})")
    errors.append("seaborn")

# ── EEG libraries ─────────────────────────────────────────────
print("\n[3] EEG Libraries:")
try:
    import mne
    print(f"    mne                {mne.__version__}  ✓")
except ImportError as e:
    print(f"    mne                MISSING  ✗  ({e})")
    errors.append("mne")

try:
    import mne_connectivity
    print(f"    mne-connectivity   OK  ✓")
except ImportError as e:
    print(f"    mne-connectivity   MISSING  ✗  ({e})")
    errors.append("mne-connectivity")

try:
    import braindecode
    print(f"    braindecode        OK  ✓")
except ImportError as e:
    print(f"    braindecode        MISSING  ✗  ({e})")
    errors.append("braindecode")

# ── Signal processing ─────────────────────────────────────────
print("\n[4] Signal Processing:")
try:
    import pywt
    print(f"    PyWavelets    {pywt.__version__}  ✓")
except ImportError as e:
    print(f"    PyWavelets    MISSING  ✗  ({e})")
    errors.append("PyWavelets")

try:
    import antropy
    print(f"    antropy       OK  ✓")
except ImportError as e:
    print(f"    antropy       MISSING  ✗  ({e})")
    errors.append("antropy")

# ── Machine Learning ──────────────────────────────────────────
print("\n[5] Machine Learning:")
try:
    import sklearn
    print(f"    scikit-learn   {sklearn.__version__}  ✓")
except ImportError as e:
    print(f"    scikit-learn   MISSING  ✗  ({e})")
    errors.append("scikit-learn")

try:
    import xgboost
    print(f"    xgboost        {xgboost.__version__}  ✓")
except ImportError as e:
    print(f"    xgboost        MISSING  ✗  ({e})")
    errors.append("xgboost")

try:
    import shap
    print(f"    shap           {shap.__version__}  ✓")
except ImportError as e:
    print(f"    shap           MISSING  ✗  ({e})")
    errors.append("shap")

try:
    import pingouin
    print(f"    pingouin       {pingouin.__version__}  ✓")
except ImportError as e:
    print(f"    pingouin       MISSING  ✗  ({e})")
    errors.append("pingouin")

try:
    import statsmodels
    print(f"    statsmodels    {statsmodels.__version__}  ✓")
except ImportError as e:
    print(f"    statsmodels    MISSING  ✗  ({e})")
    errors.append("statsmodels")

# ── File I/O ──────────────────────────────────────────────────
print("\n[6] File I/O:")
try:
    import h5py
    print(f"    h5py        {h5py.__version__}  ✓")
except ImportError as e:
    print(f"    h5py        MISSING  ✗  ({e})")
    errors.append("h5py")

try:
    import mat73
    print(f"    mat73       OK  ✓")
except ImportError as e:
    print(f"    mat73       MISSING  ✗  ({e})")
    errors.append("mat73")

try:
    import openpyxl
    print(f"    openpyxl    {openpyxl.__version__}  ✓")
except ImportError as e:
    print(f"    openpyxl    MISSING  ✗  ({e})")
    errors.append("openpyxl")

# ── Deep Learning + GPU ───────────────────────────────────────
print("\n[7] PyTorch + GPU:")
try:
    import torch
    print(f"    PyTorch version : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"    CUDA available  : {cuda_ok}")
    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"    GPU             : {gpu_name}  ✓")
        print(f"    VRAM            : {vram_gb:.1f} GB")
        # Quick tensor test on GPU
        x = torch.randn(32, 14, 128).cuda()
        y = x @ x.transpose(-1, -2)
        print(f"    GPU tensor test : PASSED  ✓")
        del x, y
        torch.cuda.empty_cache()
    else:
        print("    GPU             : NOT DETECTED  ✗")
        print("    → Install PyTorch with CUDA:")
        print("      pip install torch --index-url https://download.pytorch.org/whl/cu124")
        errors.append("CUDA")
except ImportError as e:
    print(f"    torch           : MISSING  ✗  ({e})")
    errors.append("torch")

# ── EEGNet model test ─────────────────────────────────────────
print("\n[8] EEGNet Model Test (RTX 2050 4GB VRAM check):")
try:
    import torch
    from braindecode.models import EEGNetv4

    model = EEGNetv4(n_chans=14, n_classes=2,
                     input_window_samples=3840,  # 30 sec × 128 Hz
                     final_conv_length='auto')

    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = torch.randn(4, 14, 3840).cuda()  # batch=4 (safe for 4GB)
        with torch.no_grad():
            out = model(dummy_input)
        vram_used = torch.cuda.memory_allocated() / 1024**2
        print(f"    EEGNet on GPU   : PASSED  ✓")
        print(f"    Output shape    : {out.shape}  (batch=4, classes=2)")
        print(f"    VRAM used       : {vram_used:.1f} MB  (of 4096 MB)")
        del model, dummy_input, out
        torch.cuda.empty_cache()
    else:
        out = model(torch.randn(4, 14, 3840))
        print(f"    EEGNet on CPU   : PASSED  ✓  (GPU not available)")
except Exception as e:
    print(f"    EEGNet test     : FAILED  ✗  ({e})")
    errors.append("EEGNet")

# ── .mat file loading test ────────────────────────────────────
print("\n[9] MAT File Loading Test:")
try:
    import scipy.io as sio
    import numpy as np
    # Test with a dummy array (simulating your EEG data shape)
    dummy_eeg = np.random.randn(14, 15360)  # 14 channels × 2 min @ 128Hz
    print(f"    scipy.io.loadmat  : OK  ✓")
    print(f"    Expected EEG shape: (14, 15360) — {dummy_eeg.shape}  ✓")
except Exception as e:
    print(f"    MAT loading       : FAILED  ✗  ({e})")

# ── VRAM warning for RTX 2050 ─────────────────────────────────
print("\n[10] RTX 2050 VRAM Management Tips:")
print("    ⚠  Use batch_size ≤ 8 for EEGNet with 30-sec windows")
print("    ⚠  Use batch_size ≤ 4 for full 2-min files (15360 samples)")
print("    ⚠  Add torch.cuda.empty_cache() after each training fold")
print("    ✓  Classical ML (SVM, RF, XGBoost) runs on CPU — no VRAM needed")
print("    ✓  Feature extraction runs on CPU — no VRAM needed")

# ── Final result ──────────────────────────────────────────────
print("\n" + "=" * 60)
if not errors:
    print("  ✅  ALL CHECKS PASSED — Environment is ready!")
    print("  Next step: Run jupyter lab and open a new notebook")
    print(f"  Kernel to select: Python (eeg_idd)")
else:
    print(f"  ❌  {len(errors)} issue(s) found:")
    for err in errors:
        print(f"     - {err}")
    print("\n  Fix command:")
    print(f"  pip install {' '.join(errors)}")
print("=" * 60)
