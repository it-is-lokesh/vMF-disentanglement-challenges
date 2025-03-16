# Challenges of Decomposing Tools in Surgical Scenes Through Disentangling The Latent Representations

<img src="assets/concept_figure.svg" />

This repository contains the PyTorch implementation of the network described in the paper [Challenges of Decomposing Tools in Surgical Scenes Through Disentangling The Latent Representations](https://openreview.net/forum?id=vwDshzzBrl&referrer=%5Bthe%20profile%20of%20Sai%20Lokesh%20Gorantla%5D(%2Fprofile%3Fid%3D~Sai_Lokesh_Gorantla1)) presented at the workshop ICBINB at ICLR, 2025.

## Dataset
The proposed approach was tested on Cholec80 dataset which can be downloaded by following the steps listed at [https://github.com/CAMMA-public/TF-Cholec80](https://github.com/CAMMA-public/TF-Cholec80).

## Preprocessing
After downloading the Cholec80 dataset, modify the path to the video folders in the files [`train.txt`](data/train.txt) and [`val.txt`](data/val.txt). Next, modify the path to annotations folder in the config files in `stage_1` and `stage_2` folders.

Install the necessary python packages by running

`pip install -r requirements.txt`

## Training
