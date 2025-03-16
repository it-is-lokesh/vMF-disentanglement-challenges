from config import config
from preprocess import preprocess
from helper import ClusterLoss, Cosine_Similarity, getVmfKernels
import torch.nn as nn
from torchvision.transforms import *
import random
from tqdm import tqdm as tq
import torch
import os
from torch.utils import data
from torch.backends import cudnn
import vMFNet
import numpy as np
from termcolor import colored
import json
cudnn.benchmark = True

cfg = config()

# For Reproducability
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    # Initialize image transforms and load the train and val datasets
    img_transforms = Compose([ToTensor(), Resize((256, 256), antialias=True)])

    train = preprocess(cfg.train_path, path_to_annotations=cfg.annotations, transform_images=img_transforms)
    test = preprocess(cfg.val_path, path_to_annotations=cfg.annotations, transform_images=img_transforms)

    train_dl = data.DataLoader(train, batch_size=cfg.batch_size,
                               num_workers=cfg.nworkers, pin_memory=True, shuffle=cfg.shuffle)
    test_dl = data.DataLoader(test, batch_size=cfg.batch_size,
                              num_workers=cfg.nworkers, pin_memory=True, shuffle=cfg.shuffle)

    # Initialize the model
    model = vMFNet.CompCSD()
    vMFNet.initialize_weights(model, cfg.weight_init)
    model = torch.nn.DataParallel(model, device_ids=[0]).to(cfg.device)
    pretrained = torch.load(cfg.stage1_checkpt, map_location=cfg.device)['model_state']
    model_state = model.module.state_dict()
    for i in model_state:
        model_state[i]=pretrained[i]
    model.module.load_state_dict(model_state)
    model.module.load_vmf_kernels(cfg.vmf_path)

    # Initialize the optimizer and step learnign rate decay
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.step_size, gamma=cfg.lr_decay_rate)

    # Resume training from checkpoint
    start_epoch = 0
    model_file = '.'.join([cfg.model_path, 'model'])
    if os.path.exists(model_file):
        loaded_file = torch.load(model_file)
        model.load_state_dict(loaded_file['model_state'])
        optimizer.load_state_dict(loaded_file['optim_state'])
        scheduler.load_state_dict(loaded_file['scheduler_state'])
        start_epoch = loaded_file['epoch']+1
        print('{tag} resuming from saved model'.format(
            tag=colored('[Saving]', 'red')))
        del loaded_file

    logs = []
    log_file = '.'.join([cfg.model_path, 'log'])
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)

    # Initialize the loss functions
    l2_distance = nn.MSELoss().to(cfg.device)
    l1_distance = nn.L1Loss().to(cfg.device)
    tlk_criterion = Cosine_Similarity()
    cluster_loss = ClusterLoss()

    for epoch in range(start_epoch, cfg.epochs):
        print(f"Epoch: {epoch}/{cfg.epochs}")

        model.train()

        avg_t_loss = [0]*3

        print("Training Loop!")
        for i, I in enumerate(tq(train_dl, bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')):
            I = I.to(cfg.device)

            optimizer.zero_grad()
            Ih, features, kernels = model(I)
            kernels = kernels[:64]

            rec_loss = l1_distance(Ih, I)
            clu_loss = cluster_loss(
                features.detach(), kernels) / features.shape[0]
            tlk_loss = tlk_criterion(kernels[32:], kernels[:32].detach())

            tot_loss = rec_loss + clu_loss + 0.001*tlk_loss
            tot_loss.backward()
            optimizer.step()

            losses = [rec_loss.item(), clu_loss.item(), tlk_loss.item()]
            for j in range(len(losses)):
                avg_t_loss[j] += (losses[j] - avg_t_loss[j])/(i+1)

        train_log = {
            "avg_rec_loss_t": avg_t_loss[0],
            "avg_clu_loss_t": avg_t_loss[1],
            "avg_tlk_loss_t": avg_t_loss[2],
        }

        model.eval()
        avg_v_loss = [0]*3

        with torch.no_grad():
            print("Validation Loop!")

            for i, (I) in enumerate(tq(test_dl, bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')):
                I = I.to(cfg.device)

                optimizer.zero_grad()
                Ih, features, kernels = model(I)
                kernels = kernels[:64]

                rec_loss = l1_distance(Ih, I)
                clu_loss = cluster_loss(
                    features.detach(), kernels) / features.shape[0]
                tlk_loss = tlk_criterion(kernels[32:], kernels[:32].detach())

                losses = [rec_loss.item(), clu_loss.item(), tlk_loss.item()]
                for j in range(len(losses)):
                    avg_v_loss[j] += (losses[j] - avg_v_loss[j])/(i+1)

        val_log = {
            "avg_rec_loss_v": avg_v_loss[0],
            "avg_clu_loss_v": avg_v_loss[1],
            "avg_tlk_loss_v": avg_v_loss[2],
        }

        scheduler.step()

        logs.append(
            {
                "epoch": epoch,
                "train": train_log,
                "val": val_log,
                "lr": optimizer.param_groups[0]['lr']
            }
        )

        # Save model after every epoch
        with open(log_file, 'w') as f:
            json.dump(logs, f)

        print(colored('[Saving] model saved to {}'.format(model_file), 'red'))
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        }, os.path.abspath(model_file))


if __name__ == '__main__':
    main()
