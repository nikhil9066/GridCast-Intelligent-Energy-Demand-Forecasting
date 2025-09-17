import kagglehub
import pandas as pd
import os
import shutil

def download_pjme_data():
    """Download PJME Hourly Energy Consumption Dataset from Kaggle"""

    # Download latest version
    path = kagglehub.dataset_download("robikscube/hourly-energy-consumption")
    print(f"Path to dataset files: {path}")

    # Define target directory
    target_dir = "../../data/raw"
    os.makedirs(target_dir, exist_ok=True)

    # Copy all CSV files to our data/raw directory
    for file in os.listdir(path):
        if file.endswith('.csv'):
            src = os.path.join(path, file)
            dst = os.path.join(target_dir, file)
            shutil.copy2(src, dst)
            print(f"Copied {file} to {dst}")

    return target_dir

if __name__ == "__main__":
    download_pjme_data()