import os

# ── Anchor all paths to this file (sts/config.py) ────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))   # sts/

IMAGE_SIZE = 224
EMBED_DIM  = 256

DATASET_PATH = os.path.join(r"C:\Users\user\Desktop\dataset", "Soft-tissue-Sarcoma")
CSV_PATH     = os.path.join(_REPO_ROOT, "data", "index.csv")

N_WAY  = 3
K_SHOT = 1
Q_QUERY = 1

EPISODES_PER_EPOCH = 100
EPOCHS = 10
LR     = 1e-4