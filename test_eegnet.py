from braindecode.models import EEGNet
import torch

# braindecode v1.12+: n_outputs instead of n_classes
model = EEGNet(n_chans=14, n_outputs=2, n_times=3840).cuda()
dummy = torch.randn(4, 14, 3840).cuda()

with torch.no_grad():
    out = model(dummy)

vram = torch.cuda.memory_allocated() / 1024**2
print(f"Output shape : {out.shape}")
print(f"VRAM used    : {vram:.1f} MB")
print("EEGNet PASSED")
