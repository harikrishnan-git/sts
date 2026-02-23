# 🧠 Few-Shot Learning Image Classification

A PyTorch-based Few-Shot Learning framework for image classification using episodic training.  
This project is designed to run easily by simply preparing `data/index.csv` and executing `main.py`.

---

## 🚀 Features

- Episodic Training (N-Way K-Shot)
- Works with Limited Data
- CSV-based Dataset Loader
- Modular Model Architecture (ViT / CNN)
- Training + Evaluation Pipeline
- Automatic Model Checkpoint Saving
- Confusion Matrix & Classification Report

---

## 📂 Project Structure
project/
│
├── data/
│ ├── Soft-tissue-sarcoma/ # Dataset images (class folders inside)
│ └── index.csv # Image paths and labels
│
├── models/ # Model architectures
├── utils/ # Helper functions
├── weights/ # Saved model checkpoints
│
├── main.py # Main execution file
└── README.md
