# UASTINet
Mural inpainting
Official implementation of "UASTINet: Uncertainty-Aware Joint Structure-Texture Inpainting for Dunhuang Murals".

## 1. Environment
The code was tested under the following environment:

- Python 3.9.23
- PyTorch 2.7.1
- Torchvision 0.22.1
- Torchaudio 2.7.1
- PyTorch Lightning 2.5.2
- OpenCV 4.11.0
- NumPy 1.26.4
- scikit-image 0.24.0
- LPIPS 0.1.4
- pyiqa 0.1.15

## 2. Dataset Preparation
This project uses two Dunhuang mural datasets:

- DhMurals-Inpainting
- MuralDH

The current implementation reads data from fixed train/validation/test folders. Please organize the datasets as follows:

```text
datasets/
├── DhMurals-Inpainting/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
└── MuralDH/
    ├── train/
    │   ├── images/
    │   └── masks/
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
```

The binary mask follows the convention used in the implementation:
```text
    mask = 1: missing/damaged region
    mask = 0: known/valid region
```

For reproducibility, the train/validation/test folders should follow the fixed data split reported in the paper. No additional .txt split files are required by the current implementation.


## 3. Checkpoints
The `taming/` directory contains the VQGAN implementation used by SUPNet. The corresponding VQGAN configuration and pretrained checkpoint should be placed in the `ckpt/` directory:
    ckpt/
    ├── dunhuang_vqgan.yaml
    └── epoch=vq.ckpt
 
Model checkpoints generated during training will be saved under the experiment directory:
checkpoints/
└── UASTINet/
    ├── latest_net_CSA.pth
    ├── latest_net_ES.pth
    ├── latest_net_ET.pth
    ├── latest_net_SGTFM.pth
    ├── latest_net_G.pth
    ├── latest_net_D.pth
    ├── latest_net_fuse_s.pth
    └── latest_net_fuse_t.pth

## 4. Training
To train UASTINet, run:
    python train.py \
  --model UASTINet \
  --name UASTINet \
  --dataroot datasets/DhMurals-Inpainting \
  --batchSize 32 \
  --lr 5e-5

For MuralDH, replace the dataset path:
    python train.py \
  --model UASTINet \
  --name UASTINet_MuralDH \
  --dataroot datasets/MuralDH \
  --batchSize 32 \
  --lr 5e-5

During training, the model follows the main pipeline described in the paper:
    CSA-MSPCNN enhancement
    → SUPNet structural prediction and uncertainty estimation
    → TFGCNet uncertainty-aware texture propagation
    → SGTFM structure-guided texture fusion
    → Decoder-based mural restoration

## 5. Testing
To test a trained model, run:
    python test.py \
  --model UASTINet \
  --name UASTINet \
  --dataroot datasets/DhMurals-Inpainting \
  --which_iter latest \
  --how_many 314

For MuralDH:
    python test.py \
  --model UASTINet \
  --name UASTINet_MuralDH \
  --dataroot datasets/MuralDH \
  --which_iter latest \
  --how_many 200

## 6. Evaluation

The paper reports PSNR, RMSE, SSIM, CII, LPIPS, BRISQUE, FSIM, ΔTEN, and ΔENT for quantitative evaluation.

After testing, the restored images are saved in the results directory. These results can be evaluated using standard image-quality assessment tools or customized evaluation scripts.

Please ensure that the predicted images and ground-truth images are matched by their original filenames before computing the metrics.

## 7. Repository Structure

The main implementation is under `model/`. Specifically, `csa_mspcnn.py` implements CSA-MSPCNN, `texture_graph_module.py` implements TFGCNet, `network.py` contains SUPNet, SGTFM, decoder and discriminator definitions, and `UASTINet_model.py` defines the full training and inference pipeline.

## 8. Reproducibility Notes

The implementation follows Algorithm 2 of the revised manuscript. To reproduce the reported results, please use the same dataset organization, mask convention, input resolution, and training settings described in the paper.

In our experiments, images are resized and center-cropped to 256 × 256, and UASTINet is trained with a total batch size of 32 for 500 epochs.
