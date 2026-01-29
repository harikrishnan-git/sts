from train_one_fold import train_one_fold
import os

CV_DIR = "cv"
EPISODES = 5
FOLDS = 5

for ep in range(EPISODES):
    for fold in range(FOLDS):
        print(f"\nTraining Episode {ep} Fold {fold}")

        train_csv = f"{CV_DIR}/episode_{ep}/train_fold_{fold}.csv"
        val_csv   = f"{CV_DIR}/episode_{ep}/val_fold_{fold}.csv"

        train_one_fold(train_csv, val_csv)
