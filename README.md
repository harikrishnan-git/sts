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

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/harikrishnan-git/sts.git
cd sts
```

---

### 2️⃣ (Optional) Create Virtual Environment

```bash
python -m venv venv
```

#### Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / Mac:**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

If `requirements.txt` exists:

```bash
pip install -r requirements.txt
```

Otherwise:

```bash
pip install torch torchvision pandas scikit-learn matplotlib
```

## ▶️ How to Run

Simply execute:

```bash
python main.py
```

---

### What the Script Does

The script will:

- Load dataset from `data/index.csv`
- Create episodic batches
- Train the few-shot model
- Evaluate performance
- Save trained weights in `weights/`

## 🧪 Configuration (Inside `main.py`)

Example:

```python
N_WAY = 3
K_SHOT = 5
EPISODES = 200
```

Modify these values based on your dataset size and number of classes.

---

## 📈 Evaluation & Output

### Step 1: Train the Model

```bash
python main.py
```

This will train the few-shot model and save the trained weights inside the `weights/` directory.

---

### Step 2: Run Evaluation

After training is complete, run:

```bash
python eval/evaluate.py
```

---

### Evaluation Output

The following metrics will be displayed:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score
- Loaded model checkpoint details
