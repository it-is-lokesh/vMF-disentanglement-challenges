import numpy as np
from config import config

cfg = config()

vmf_kernels = np.random.uniform(low=0, high=1, size=(32, 16))
# np.save(cfg.vmf_path, vmf_kernels)
np.save('tmp.npy', vmf_kernels)