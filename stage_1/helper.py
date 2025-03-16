import torch.nn as nn
import sys
import numpy as np
import torch
from config import config
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image

cfg = config()
device = 'cuda' if torch.cuda.is_available() and cfg.gpu else 'cpu'


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

        return loss


class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(0)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x.squeeze(3).squeeze(2)
        y = y.squeeze(3).squeeze(2)

        cosine_sim_matrix = F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=-1)
        num_vectors = x.size(0)
        indices = torch.arange(num_vectors).repeat(2, 1)
        cosine_sim_matrix[indices[0], indices[1]] = 0
        average_cosine_similarity = cosine_sim_matrix.mean()

        return average_cosine_similarity


def getVmfKernels(dict_dir):
    vc = np.load(dict_dir, allow_pickle=True)
    vc = vc[:, :, np.newaxis, np.newaxis]
    vc = torch.from_numpy(vc).type(torch.FloatTensor)
    return vc
