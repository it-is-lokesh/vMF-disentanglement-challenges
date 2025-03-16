import torch.nn as nn
import numpy as np
import torch
from torch.nn import functional as F

class ClusterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        eps = 1e-9
        loss = 0
        y = y.squeeze(3).squeeze(2)
        for i in x:
            i = i.permute(1, 2, 0)
            i = i.reshape(-1, i.shape[2])
            length = i.shape[0]

            m = i / (i.norm(dim=1, keepdim=True) + eps)
            n = (y / (y.norm(dim=1, keepdim=True) + eps)).transpose(0, 1)

            z = m.mm(n)

            z = z.max(dim=1)[0]
            loss += (1. - z).sum() / float(length)

        # loss /= x.shape[0]
        return loss
    

class Cosine_Similarity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x.squeeze(3).squeeze(2)
        y = y.squeeze(3).squeeze(2)

        cosine_sim_matrix = F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=-1)
        average_cosine_similarity = cosine_sim_matrix.mean()
        return average_cosine_similarity


def getVmfKernels(dict_dir):
    vc = np.load(dict_dir, allow_pickle=True)
    vc = vc[:, :, np.newaxis, np.newaxis]
    vc = torch.from_numpy(vc).type(torch.FloatTensor)
    return vc
