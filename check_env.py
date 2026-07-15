#!/usr/bin/env python3
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('Device count:', torch.cuda.device_count())
