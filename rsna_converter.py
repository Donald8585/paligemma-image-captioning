#!/usr/bin/env python3
"""
RSNA DICOM to PNG Converter + Caption CSV Generator
Converts RSNA cervical spine DICOMs to PNGs and creates train/val CSVs for PaliGemma.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import pydicom
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def convert_dicom_to_png(dicom_path, output_path, window_center=40, window_width=400):
    """Convert DICOM to PNG with windowing for bone visualization."""
    try:
        dcm = pydicom.dcmread(dicom_path)
        img = dcm.pixel_array.astype(float)

        # Apply windowing for bone
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)

        # Normalize to 0-255
        img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        # Save as PNG
        Image.fromarray(img).save(output_path)
        return True
    except Exception as e:
        print(f"Error converting {dicom_path}: {e}")
        return False


def create_caption(row):
    """Generate caption from fracture labels."""
    fractures = []
    for vertebra in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        col = f'{vertebra}_fracture'
        if col in row and row[col] == 1:
            fractures.append(vertebra)

    if fractures:
        return f"cervical spine fracture at {', '.join(fractures)}"
    else:
        return "no cervical spine fracture detected"


def main():
    # Paths
    input_dir = '/kaggle/input/rsna-2022-cervical-spine-fracture-detection'
    train_csv_path = f'{input_dir}/train.csv'
    dicom_dir = f'{input_dir}/train_images'

    output_dir = '/kaggle/working/rsna_processed'
    output_images_dir = f'{output_dir}/images'
    os.makedirs(output_images_dir, exist_ok=True)

    # Load labels
    print("Loading train.csv...")
    df = pd.read_csv(train_csv_path)

    # Sample subset for quick testing (remove this line for full dataset)
    print("Sampling 500 studies for quick test...")
    df = df.sample(n=min(500, len(df)), random_state=42)

    print(f"Processing {len(df)} studies...")

    processed_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        study_id = row['StudyInstanceUID']

        # Find DICOM files for this study
        study_path = os.path.join(dicom_dir, study_id)
        if not os.path.exists(study_path):
            continue

        dicom_files = [f for f in os.listdir(study_path) if f.endswith('.dcm')]

        # Take middle slice (representative view)
        if len(dicom_files) == 0:
            continue

        mid_idx = len(dicom_files) // 2
        dicom_file = sorted(dicom_files)[mid_idx]

        dicom_path = os.path.join(study_path, dicom_file)
        png_filename = f"{study_id}.png"
        png_path = os.path.join(output_images_dir, png_filename)

        # Convert DICOM to PNG
        if convert_dicom_to_png(dicom_path, png_path):
            caption = create_caption(row)
            processed_data.append({
                'image_path': png_filename,
                'text': caption
            })

    print(f"\nSuccessfully converted {len(processed_data)} images")

    # Create DataFrame
    processed_df = pd.DataFrame(processed_data)

    # Split train/val (80/20)
    train_df, val_df = train_test_split(processed_df, test_size=0.2, random_state=42)

    # Save CSVs
    train_csv_out = f'{output_dir}/train.csv'
    val_csv_out = f'{output_dir}/val.csv'

    train_df.to_csv(train_csv_out, index=False)
    val_df.to_csv(val_csv_out, index=False)

    print(f"\n✅ Created {train_csv_out} with {len(train_df)} samples")
    print(f"✅ Created {val_csv_out} with {len(val_df)} samples")
    print(f"✅ Images saved to {output_images_dir}")

    # Show sample
    print("\nSample captions:")
    print(train_df.head())


if __name__ == "__main__":
    main()
