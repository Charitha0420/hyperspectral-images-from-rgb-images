【**2023.12.12**】 The HPRN model havs been updated.

# HPRN: Holistic Prior-embedded Relation Network for Spectral Super-Resolution

Chaoxiong Wu, Jiaojiao Li, Rui Song, Yunsong Li, Qian Du. [Paper](https://ieeexplore.ieee.org/abstract/document/10091189). 

## HPRN framework
- Network architecture of the proposed network.
![](./Figures/HPRN.jpg)

## Citation
- If the the work or the code is helpful, please cite the paper:  

@article{wu2023hprn,  
  title={Hprn: Holistic prior-embedded relation network for spectral super-resolution},  
  author={Wu, Chaoxiong and Li, Jiaojiao and Song, Rui and Li, Yunsong and Du, Qian},  
  journal={IEEE Transactions on Neural Networks and Learning Systems},  
  year={2023},  
  publisher={IEEE}  
}
 
# Reconstructing Hyperspectral Images from RGB Images using HPRN

## Overview

This project focuses on **Spectral Super-Resolution (SSR)**, the task of reconstructing a high-dimensional Hyperspectral Image (HSI) from a conventional RGB image.

A standard RGB image contains only three channels (Red, Green, and Blue), whereas a hyperspectral image contains multiple narrow spectral bands that provide detailed information about the materials present in a scene.

The project implements the **Holistic Prior-Embedded Relation Network (HPRN)**, a deep learning architecture designed to reconstruct hyperspectral information from RGB images by exploiting spatial and spectral relationships.

The HPRN model was implemented using **Python and PyTorch** and evaluated on the **CAVE** and **Harvard** hyperspectral datasets.

---

## Problem Statement

Hyperspectral cameras provide significantly more spectral information than conventional RGB cameras, but they are generally more expensive and complex.

The goal of Spectral Super-Resolution is therefore:

```text
RGB Image (3 Channels)
        ↓
     HPRN Model
        ↓
Hyperspectral Image (31 Channels)

The problem is highly ill-posed because many different hyperspectral spectra can potentially correspond to the same RGB observation.

HPRN addresses this challenge by incorporating multiple types of prior information and learning relationships in both the spatial and spectral domains.

Objectives

The main objectives of this project were:

Study and understand the HPRN architecture.
Understand the problem of RGB-to-Hyperspectral image reconstruction.
Implement the HPRN architecture using Python and PyTorch.
Implement the major HPRN components:
Multiresidual Relation Blocks (MRB)
Semantic-driven Spatial Relation Module (SSRM)
Transformer-based Channel Relation Module (TCRM)
Train and evaluate the model on the CAVE and Harvard datasets.
Generate reconstructed 31-channel hyperspectral images.
Evaluate reconstruction quality using PSNR, SSIM, and SAM.
Perform qualitative comparison between ground-truth and reconstructed hyperspectral images.
HPRN Architecture

The implemented HPRN architecture consists of several major components.

1. Shallow Feature Extraction

The RGB input is represented as:

(B, 3, H, W)

A 3×3 convolution is first applied to project the RGB image into a higher-dimensional feature space.

The resulting shallow feature representation is denoted as F0.

2. Multiresidual Relation Blocks (MRB)

The MRBs form the main backbone of the network.

Each MRB consists of:

Input
  ↓
3×3 Convolution
  ↓
ReLU
  ↓
3×3 Convolution
  ↓
Residual Addition

Residual connections help preserve important low-frequency information from the input while allowing the network to learn additional high-frequency details.

A global residual connection also carries the initial shallow features to later stages of the network.

3. Semantic-driven Spatial Relation Module (SSRM)

The SSRM is used to capture spatial and semantic relationships within the image.

The module:

Performs global average pooling.
Uses an MLP to process the extracted representation.
Applies a sigmoid activation to generate attention weights.
Uses the resulting attention information to refine spatial features.

This allows the network to capture relationships between pixels belonging to similar objects or materials, even when they are spatially separated.

4. Transformer-based Channel Relation Module (TCRM)

The TCRM models relationships between spectral channels.

The feature map is transformed into a sequence where spectral channels act as tokens.

Query (Q), Key (K), and Value (V) representations are generated and self-attention is applied to model relationships between channels.

The resulting features are passed through a feed-forward network and reshaped back into image format.

This enables the model to capture long-range dependencies between different spectral bands.

5. Fusion and Reconstruction

The spatially refined features from SSRM and spectrally refined features from TCRM are combined.

SSRM Features ─────┐
                   ├──→ Feature Fusion → Reconstruction → 31-channel HSI
TCRM Features ─────┘

A 1×1 convolution is used for feature fusion, followed by a reconstruction layer that produces the final:

(B, 31, H, W)

hyperspectral output.

The reconstructed HSI covers the spectral range of approximately 400 nm to 700 nm.

Datasets
CAVE Dataset

The CAVE dataset contains 32 hyperspectral images consisting of indoor scenes such as colorful objects, food, and artificial flowers.

The controlled indoor environment provides a useful setting for evaluating spectral reconstruction.

For this project, the images were normalized to the range:

[0, 1]

The CAVE images were divided into overlapping 64×64 patches to increase the number of training samples and reduce GPU memory requirements.

Harvard Dataset

The Harvard dataset contains 27 hyperspectral images containing both indoor and outdoor scenes.

Unlike the controlled CAVE dataset, Harvard contains more realistic scenes with natural illumination.

For the Harvard experiments, full 512×512 images were used rather than patch-based training.

Data Processing

The main preprocessing steps were:

Load RGB and hyperspectral images.
Normalize pixel values to the range [0, 1].
Generate training patches for the CAVE dataset.
Keep the test data separate from training data.
Use a fixed random seed for reproducible data splitting.
CAVE
Original Image
      ↓
Normalization
      ↓
64 × 64 Overlapping Patches
      ↓
Train / Test Split
Harvard
Original 512 × 512 Image
          ↓
     Normalization
          ↓
     Train / Test Split
Training

The model was trained using PyTorch on the Kaggle platform with an NVIDIA T4 GPU.

Training Configuration
Parameter	Value
Framework	PyTorch
GPU	NVIDIA T4
Loss Function	L1 Loss
Optimizer	Adam
Number of Epochs	100
CAVE Patch Size	64 × 64
Harvard Image Size	512 × 512
Output Channels	31
Loss Function

The model uses L1 Loss (Mean Absolute Error):

L1 Loss = mean(|Predicted HSI - Ground Truth HSI|)

The loss measures the difference between the reconstructed hyperspectral image and the ground-truth hyperspectral image.

Evaluation Metrics

Three metrics were used to evaluate the reconstructed hyperspectral images.

PSNR

Peak Signal-to-Noise Ratio (PSNR) measures the overall reconstruction quality.

Higher PSNR generally indicates lower reconstruction error.

SSIM

Structural Similarity Index Measure (SSIM) evaluates structural similarity between the reconstructed and ground-truth images.

It considers characteristics such as:

Edges
Local structures
Contrast
Spatial information
SAM

Spectral Angle Mapper (SAM) evaluates the spectral similarity between the reconstructed and ground-truth hyperspectral pixels.

A lower SAM value indicates better spectral similarity.

Results

The implemented model was evaluated on both the CAVE and Harvard datasets.

Dataset	PSNR (dB)	SSIM	SAM (degrees)
CAVE	33.3436	0.9345	9.7636
Harvard	34.8635	0.7708	62.4899
CAVE Results

The CAVE dataset achieved:

PSNR : 33.3436 dB
SSIM : 0.9345
SAM  : 9.7636°

The patch-based training strategy allowed the model to learn local structures and textures effectively.

Harvard Results

The Harvard dataset achieved:

PSNR : 34.8635 dB
SSIM : 0.7708
SAM  : 62.4899°

The Harvard experiment used full 512×512 images, which resulted in significantly higher memory requirements and required smaller batch sizes on the available GPU resources.

Visualization

Since an HSI contains 31 spectral channels, direct visualization of the complete hyperspectral cube is difficult.

For visualization, different approaches were used.

CAVE

Specific hyperspectral bands were selected:

Red   → Band 25
Green → Band 15
Blue  → Band 5

This allows direct inspection of selected spectral bands.

Harvard

The dataset's Camera Spectral Sensitivity (CSS) information was used to combine the reconstructed 31 spectral bands into an RGB representation.

This provides a more realistic camera-like visualization of the reconstructed hyperspectral image.

Qualitative Evaluation

The project also compares:

Input RGB image
Ground-truth HSI
Predicted HSI
Ground-truth spectral bands
Predicted spectral bands
Absolute error maps

These visualizations help analyze reconstruction quality and identify spatial or spectral reconstruction errors.

Project Workflow
                RGB Image
                    │
                    ▼
            Data Preprocessing
                    │
                    ▼
              Normalization
                    │
                    ▼
          Patch Generation
          (CAVE Dataset)
                    │
                    ▼
        Shallow Feature Extraction
                    │
                    ▼
       Multiresidual Relation Blocks
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
        SSRM                 TCRM
          │                   │
          │ Spatial           │ Spectral
          │ Relations         │ Relations
          └─────────┬─────────┘
                    ▼
              Feature Fusion
                    │
                    ▼
             Reconstruction
                    │
                    ▼
          31-Channel HSI
                    │
                    ▼
          PSNR / SSIM / SAM
Technologies Used
Python
PyTorch
NumPy
OpenCV
Matplotlib
Kaggle
NVIDIA CUDA / GPU
Key Contributions

The major work carried out in this project includes:

Studied the HPRN research architecture for Spectral Super-Resolution.
Implemented the HPRN network using PyTorch.
Implemented MRB, SSRM, and TCRM components.
Developed preprocessing and patch-generation pipelines.
Trained the model on CAVE and Harvard datasets.
Generated 31-channel hyperspectral reconstructions.
Evaluated the model using PSNR, SSIM, and SAM.
Performed qualitative visualization of reconstructed hyperspectral images.
Investigated the effect of patch-based versus full-image training under limited GPU resources.
Limitations

The experiments were performed using limited computational resources available through the free Kaggle GPU environment.

The models were trained for 100 epochs, which is considerably shorter than a full-scale research training setup.

The Harvard experiment was particularly constrained by GPU memory because the model was trained on full-resolution 512×512 images.

These computational limitations affect the comparison of the obtained results with results reported in the original HPRN paper.

Future Work

Several improvements can be explored in future versions of the project.

1. Lightweight Spectral Attention

The TCRM can be optimized using alternatives such as:

Linear Attention
Sparse Attention

This could reduce memory consumption while retaining long-range spectral relationships.

2. Multi-Scale Spatial Processing

The SSRM can potentially be extended using multi-scale or pyramid-based spatial processing to better capture both large objects and fine details.

3. Adaptive Camera Response Learning

A camera-response estimation module could be introduced to automatically estimate the spectral response characteristics of an input camera.

This could improve the generalization of the model across different cameras and imaging systems.

References
C. Wu, J. Li, R. Song, Y. Li, and Q. Du,
"HPRN: Holistic Prior-Embedded Relation Network for Spectral Super-Resolution,"
IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 8, pp. 11409–11422, Aug. 2024.
F. Yasuma, T. Mitsunaga, D. Iso, and S. K. Nayar,
"Generalized Assorted Pixel Camera: Post-Capture Control of Resolution, Dynamic Range, and Spectrum,"
IEEE Transactions on Image Processing, 2010.
A. Chakrabarti and T. Zickler,
"Statistics of Real-World Hyperspectral Images,"
CVPR, 2011.
B. Arad and O. Ben-Shahar,
"Sparse Recovery of Hyperspectral Signal from Natural RGB Images,"
ECCV, 2016.
Y. Fu, T. Zhang, Y. Zheng, D. Zhang, and H. Huang,
"Joint Camera Spectral Sensitivity Selection and Hyperspectral Image Recovery,"
ECCV, 2018.
K. He, X. Zhang, S. Ren, and J. Sun,
"Deep Residual Learning for Image Recognition,"
CVPR, 2016.
A. Vaswani et al.,
"Attention Is All You Need,"
NeurIPS, 2017.
Acknowledgements

I would like to thank Prof. Debashis Sen, Department of Electronics and Electrical Communication Engineering, IIT Kharagpur, for his guidance and support throughout this project.

I also acknowledge the support provided during the implementation and research process.

Author

Gonuguntla Charitha
B.Tech, Electronics and Electrical Communication Engineering
Indian Institute of Technology Kharagpur

Project Supervisor: Prof. Debashis Sen


