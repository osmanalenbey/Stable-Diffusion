# Diffusion Models from Scratch (with PyTorch)

This notebook presents a minimal yet fully functional implementation of a **conditional diffusion model**, built entirely from scratch using PyTorch. It covers the entire pipeline:

- Forward diffusion (adding Gaussian noise to data over time),
- Reverse diffusion (denoising using a trained model),
- Conditional generation via label injection (class-conditioning),
- Sample interpolation to demonstrate latent space structure.

The code is written for clarity and educational purposes, and targets the MNIST dataset to make the dynamics visually intuitive.


<p align="center">
  <img width="646" alt="image" src="https://github.com/user-attachments/assets/08a60686-79b4-44d9-8d7f-51d77e711e10" />
</p>


## 🔍 Key Highlights

- **Conditional Sampling**: Incorporates class labels as one-hot vectors for controlled digit generation.
- **Score-Free Training**: The model is trained to estimate the original clean signal x0 rather than the noise or score.
- **Reverse Diffusion as a Generator**: The only learnable part of the pipeline is the reverse diffuser, which is trained to undo the noising process.
- **Visualization Tools**: Includes diagrams and interpolation examples to showcase how diffusion enables meaningful transitions.

## 🖼️ Sample Outputs

- Class-conditional digit generation from noise.
- Smooth interpolations across digits with fixed or varying noise seeds.
- Training progression and qualitative improvements over epochs.

  <img width="338" alt="image" src="https://github.com/user-attachments/assets/656c22c4-16c9-4602-bf4f-c2d9eac1724a" />

## 📚 Reference

This project is inspired by the architecture and principles described in:

> **Rombach et al.**, [*High-Resolution Image Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752), 2022

While this repository implements a much simpler version, the core idea of forward noising and learned reverse denoising remains central.
