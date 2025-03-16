import numpy as np
from config import config
import torch

cfg = config()

tool_vmf_kernels = np.random.uniform(low=0, high=1, size=(32, 16))
bg_vmf_kernels = torch.load("../checkpoints/stage_1.model")['model_state']['module.conv1o1.weight'].squeeze(-1).squeeze(-1).cpu().numpy()
vmf_kernels = np.concatenate([bg_vmf_kernels, tool_vmf_kernels], axis=0)
np.save(cfg.vmf_path, vmf_kernels)
