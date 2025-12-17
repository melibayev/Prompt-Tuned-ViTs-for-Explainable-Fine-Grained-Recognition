# Prompt-Tuned Vision Transformers for Explainable Image Classification

This repository contains a simple and educational implementation of **Prompt-Tuned Vision Transformers (ViT)** for fine-grained image classification with visual explanations using **attention rollout**. The project is designed for academic purposes and focuses on clarity, efficiency, and interpretability.

---

## Project Overview

* **Task**: Fine-grained image classification
* **Dataset**: Oxford-102 Flowers
* **Model**: Vision Transformer (ViT-B/16)
* **Method**: Prompt tuning (train prompts instead of full model)
* **Explainability**: Attention rollout visualization

The main idea is to adapt a large pre-trained Vision Transformer to a new task by training only a small number of parameters, while also visualizing where the model focuses when making predictions.

---

## Key Concepts (Simple Explanation)

* **Vision Transformer (ViT)**: Splits an image into patches and processes them using attention.
* **Prompt Tuning**: Adds small trainable tokens (prompts) to guide the model without retraining everything.
* **Attention Rollout**: Combines attention maps from all transformer layers to show important image regions.

---

## Project Structure

```
├── data/                     # Dataset directory (downloaded automatically)
├── checkpoints/              # Saved model checkpoints
├── best_model.pth            # Best trained model
├── example_flower.jpg        # Example test image
├── notebook.ipynb            # Main Colab / Jupyter notebook
├── README.md                 # Project documentation
```

---

## Requirements

Install required libraries:

```bash
pip install torch torchvision timm transformers matplotlib scipy tqdm
```

The project is recommended to run on **Google Colab** with GPU support, but it also works on CPU (slower).

---

## How to Run

1. Open the notebook in Google Colab or Jupyter.
2. Install dependencies (Cell 1).
3. Run cells in order:

   * Load dataset
   * Build prompt-tuned ViT model
   * Train the model
   * Evaluate accuracy
   * Visualize attention rollout

The dataset will be downloaded automatically.

---

## Training Details

* **Loss Function**: Cross-Entropy Loss
* **Optimizer**: Adam
* **Batch Size**: 16
* **Epochs**: 3
* **Trainable Parameters**:

  * Prompt tokens
  * Classification head
* **Frozen Parameters**:

  * ViT backbone

---

## Results

* **Training Accuracy**: up to 100%
* **Test Accuracy**: ~98.7%

The model converges quickly due to the use of a pre-trained backbone and prompt tuning.

---

## Explainability

The project uses **attention rollout** to visualize model decisions:

* Shows where the model looks when predicting a class
* Produces coarse heatmaps due to patch-based processing
* Helps understand model behavior and failure cases

---

## Limitations

* Evaluated on a single dataset
* Attention maps are approximate explanations
* No direct baseline training included in code

These limitations are discussed in the final report.

---

## Authors

* **Student IDs**: 220730, 220520, 220940
* **Course**: Computer Vision
