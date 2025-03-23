from config import config
from preprocess import preprocess
from helper import *
from torchvision.transforms import Compose, ToTensor, Resize
import random
from tqdm import tqdm as tq
import torch
import os
from torch.utils import data
from torch.backends import cudnn
import vMFNet
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import json
cudnn.benchmark = True

cfg = config()

def main():
    # Initialize image transforms and load the train and val datasets
    img_transforms = Compose([ToTensor(), Resize((256, 256), antialias=True)])

    val = preprocess(cfg.val_path, path_to_annotations=cfg.annotations, transform_images=img_transforms)

    val_dl = data.DataLoader(val, batch_size=1, num_workers=1, pin_memory=False)

    # Initialize the model
    model = vMFNet.CompCSD(train_flag=False)
    vMFNet.initialize_weights(model, cfg.weight_init)
    model.load_vmf_kernels(cfg.vmf_path)
    model = torch.nn.DataParallel(model, device_ids=[0]).to(cfg.device)

    # Resume training from checkpoint
    model_file = '.'.join([cfg.model_path, 'model'])
    if os.path.exists(model_file):
        loaded_file = torch.load(model_file)
        model.module.load_state_dict(loaded_file['model_state'])
        print('{tag} resuming from saved model'.format(
            tag=colored('[Saving]', 'red')))
        del loaded_file

    if not os.path.exists('inference'):
        os.mkdir('inference')
    
    with torch.no_grad():
        print("Validation Loop!")
        for i, (I) in enumerate(tq(val_dl, bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')):
            I = I.to(cfg.device)

            if not os.path.exists(f'inference/{i}'):
                os.mkdir(f"inference/{i}")

            Ih, features, norm_vmf_activations, decoding_features = model(I)
            fig, ax = plt.subplots(1)
            ax.imshow(I[0].permute(1,2,0).cpu().numpy())
            ax.axes('off')
            fig.savefig(f"inference/{i}/inp_image.png")
            plt.close()

            fig, ax = plt.subplots(1)
            ax.imshow(Ih[0].permute(1,2,0).cpu().numpy())
            ax.axes('off')
            fig.savefig(f"inference/{i}/rec_image.png")
            plt.close()

            fig, ax = plt.subplots(4, 8)
            for j in range(32):
                ax[j//8, j%8].imshow(norm_vmf_activations[0, j], cmap='gray')
                ax[j//8, j%8].axes('off')
            fig.savefig(f"inference/{i}/disentangled_image.png")
            plt.close()


if __name__ == '__main__':
    main()
