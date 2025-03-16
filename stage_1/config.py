import os
import torch


class config:
    def __init__(self):
        self.seed = 3407
        self.gpu = True
        self.device = 'cuda' if torch.cuda.is_available() and self.gpu else 'cpu'
        self.shuffle = True
        self.pin_memory = True

        self.batch_size = 32
        self.nworkers = 16
        self.lr = 1e-3
        self.step_size = 10
        self.lr_decay_rate = 0.9
        self.epochs = 15

        self.vMF_kappa = 30
        self.weight_init = 'xavier'
        self.alpha1 = 3
        self.beta1 = 0.05

        self.base_dir = os.getcwd()
        self.expt_name = 'stage_1'
        self.vmf_path = os.path.join(self.base_dir, 'weights', 'vmf_kernels.npy')
        self.expt_path = os.path.join(self.base_dir, '../checkpoints')
        self.model_path = self.expt_path + self.expt_name
        self.train_path = os.path.join(self.base_dir, '../data/train.txt')
        self.val_path = os.path.join(self.base_dir, '../data/val.txt')
        self.annotations = '/path/to/annotations'
        if not os.path.isdir(self.expt_path):
            os.mkdir(self.expt_path)


if __name__=='__main__':
    cfg = config()
    print(cfg.model_path)
