# Deep Learning for Artist Identification & Art Style Transfer

This repository contains two deep learning projects: one for automatically identifying artists from their paintings and another for transferring artistic style from one painting to another.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset): https://drive.google.com/drive/folders/1IwOD6Jz76Fs1jo6_gccnvga3F3-4cgpM
- [Artist Identification](#artist-identification)
  - [Data Processing](#data-processing)
  - [Model Architecture & Training](#model-architecture--training)
  - [Results & Evaluation](#results--evaluation)
- [Art Style Transfer](#art-style-transfer)
  - [Methodology](#methodology)
  - [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

This repository implements two projects using deep learning techniques:

1. **Artist Identification:**  
   Develops a CNN-based pipeline to classify paintings by artist. The dataset was filtered to the 11 most represented artists (≥200 paintings each) to ensure sufficient data. The model uses a pre-trained ResNet50 with fine-tuning and class weights to address class imbalance.

2. **Art Style Transfer:**  
   Applies neural style transfer by extracting style and content features from paintings using pre-trained networks (VGG19 and InceptionV3). It then combines these features to generate a new image that blends the content of one image with the style of another.

---

## Dataset

The projects use the **Best Artworks of All Time** dataset from Kaggle:
[Best Artworks of All Time Dataset](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)

- **Artist Metadata:** Provided in `artists.csv` containing artist names and painting counts.
- **Images:** Full-sized images of artworks organized by artist folders.
- **Filtering:** For artist identification, only 11 artists with ≥200 paintings were selected to ensure robust training and address data imbalance.

---

## Artist Identification

### Data Processing

- **Filtering:**  
  From 50 artists, only the 11 with at least 200 paintings were selected to reduce computation and ensure sufficient samples per class.
  
- **Class Imbalance:**  
  Significant imbalance exists (e.g., Van Gogh has 877 paintings vs. Marc Chagall with 239). Class weights were computed and applied during training to give higher importance to underrepresented classes.

- **Preprocessing:**  
  Images are resized to 224×224 pixels and normalized to a [0,1] scale. Data augmentation (shear, horizontal & vertical flips, zoom, etc.) was applied using Keras' ImageDataGenerator.

### Model Architecture & Training

- **Base Model:**  
  A pre-trained ResNet50 (with ImageNet weights) is used as the feature extractor.  
  - The network's shallow layers are fine-tuned to capture subtle artistic styles while deeper layers remain frozen.
  
- **Custom Classification Head:**  
  - GlobalAveragePooling2D  
  - Dense layers with 256 (or 128) units with ReLU activation, Batch Normalization, and Dropout for regularization  
  - Final Dense layer with softmax activation for artist classification

- **Training Strategy:**  
  - Two-phase training:  
    1. Train the full model for an initial set of epochs with class weights.  
    2. Freeze deeper layers and fine-tune shallow layers for further epochs.
  - Optimizer: Adam (learning rate = 0.0001)  
  - Loss: Categorical or sparse categorical cross-entropy  
  - Callbacks: EarlyStopping and ReduceLROnPlateau were used to stabilize training.

### Results & Evaluation

- **Performance:**  
  - Training Accuracy: ~99%  
  - Validation Accuracy: ~85%  
- **Evaluation:**  
  - Confusion Matrix and classification report provide insights into which artists are often confused.
  - Visual predictions on random images demonstrate the model’s capability to recognize artist styles.

---

## Art Style Transfer

### Methodology

- **Objective:**  
  Generate a new image by blending the content of a base image with the style of a style image.

- **Preprocessing:**  
  - Resize images to a specified target size (e.g., 400 pixels in height) and normalize inputs.
  - Use model-specific preprocessing (e.g., VGG19 subtracts mean RGB values).

- **Feature Extraction:**  
  - Use a pre-trained network (VGG19 or InceptionV3) to extract content and style features.
  - For VGG19, the content is extracted from `block5_conv2` and style from layers `block1_conv1` to `block5_conv1`.

- **Loss Computation:**  
  - **Content Loss:** Measures the difference between the content of the base and generated image.
  - **Style Loss:** Computes differences between Gram matrices (capturing texture and style) of the style image and the generated image.
  - **Total Loss:** A weighted sum of content and style losses.

- **Optimization:**  
  - The L-BFGS algorithm is used to iteratively update the generated image to minimize the total loss.

### Implementation Details

- **Preprocessing & Deprocessing Functions:**  
  Custom functions handle image loading, resizing, and reverting the preprocessed image back to a displayable format.
  
- **Feature Extractor:**  
  A Keras model is created that outputs both style and content features.
  
- **Gradient Computation:**  
  Loss and gradients are computed using TensorFlow’s GradientTape, and the optimizer (L-BFGS) adjusts the generated image.

---

## Usage

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/yourusername/Deep_Learning_Artist_identification_and_style_transfer.git
   cd Deep_Learning_Artist_identification_and_style_transfer
   ```

2. **Install Dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Paths:**  
   Update the paths in the code for `data_dir` and `csv_file` to point to your dataset and metadata.

4. **Run Notebooks:**  
   Execute the provided Jupyter notebooks to train the artist identification model and perform art style transfer.

---

## Future Work

- Experiment with alternative architectures (e.g., EfficientNet, InceptionV3) for artist identification.
- Fine-tune additional layers and explore more aggressive data augmentation.
- Improve style transfer by normalizing losses and exploring additional style layers.
- Extend the dataset to include more artists for broader applicability.

---

This project is run on MLDS server

---
