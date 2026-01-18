#!/usr/bin/env python3
"""
PaliGemma Fine-Tuning Script
Minimal working script for fine-tuning PaliGemma on image-text pairs using LoRA.
Supports Kaggle and Colab environments.

Usage:
    python paligemma_finetune.py --train_csv /path/to/train.csv --val_csv /path/to/val.csv --image_root /path/to/images
"""

import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


class ImageTextDataset(Dataset):
    """Dataset wrapper for image-text pairs."""

    def __init__(self, csv_path: str, image_root: str, processor, max_length: int = 128):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.processor = processor
        self.max_length = max_length
        print(f"Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root, row["image_path"])
        text = row["text"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            raise

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Squeeze batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


@dataclass
class DataCollator:
    """Custom data collator for batching."""
    processor: Any

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = batch[0].keys()
        collated = {}
        for k in keys:
            collated[k] = torch.stack([example[k] for example in batch])
        return collated


def setup_model_and_processor(model_name: str, lora_r: int = 8, lora_alpha: int = 8):
    """Load processor and model with LoRA configuration."""

    print(f"Loading processor from {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)

    print(f"Loading base model from {model_name}...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return processor, model


def main(args):
    """Main training pipeline."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup model and processor
    processor, model = setup_model_and_processor(
        args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )

    # Load datasets
    print("
Loading datasets...")
    train_dataset = ImageTextDataset(args.train_csv, args.image_root, processor, args.max_length)
    val_dataset = ImageTextDataset(args.val_csv, args.image_root, processor, args.max_length)

    # Data collator
    data_collator = DataCollator(processor=processor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        bf16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=args.num_workers,
    )

    # Trainer
    print("
Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("
Starting training...")
    trainer.train()

    # Save adapter
    adapter_output_dir = args.adapter_output_dir or os.path.join(args.output_dir, "lora_adapter")
    print(f"
Saving LoRA adapter to {adapter_output_dir}...")
    model.save_pretrained(adapter_output_dir)
    processor.save_pretrained(adapter_output_dir)

    print("
✅ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune PaliGemma with LoRA on image-text pairs")

    # Data paths
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV (columns: image_path, text)")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory containing images")

    # Model config
    parser.add_argument("--model_name", type=str, default="google/paligemma-3b-mix-224", help="PaliGemma model name")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha")

    # Training config
    parser.add_argument("--output_dir", type=str, default="./paligemma-med-finetune", help="Output directory")
    parser.add_argument("--adapter_output_dir", type=str, default=None, help="LoRA adapter output directory (default: output_dir/lora_adapter)")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint frequency")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")

    args = parser.parse_args()

    # Validate paths
    assert os.path.exists(args.train_csv), f"Train CSV not found: {args.train_csv}"
    assert os.path.exists(args.val_csv), f"Val CSV not found: {args.val_csv}"
    assert os.path.exists(args.image_root), f"Image root not found: {args.image_root}"

    main(args)
