"""Fine-tune a pretrained vision-encoder-decoder model for image captioning."""

from pathlib import Path

import pandas as pd
from transformers import (
    AutoTokenizer,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
)

from src.dataset import CaptionDataset

BASE_MODEL = "nlpconnect/vit-gpt2-image-captioning"


def load_pretrained() -> tuple[VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast]:
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)
    image_processor = ViTImageProcessor.from_pretrained(BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    return model, image_processor, tokenizer


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    images_dir: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 8,
) -> tuple[Trainer, ViTImageProcessor, GPT2TokenizerFast]:
    """Fine-tune the pretrained captioning model on train_df, tracking val_df loss each epoch."""
    model, image_processor, tokenizer = load_pretrained()

    train_dataset = CaptionDataset(train_df, images_dir, image_processor, tokenizer)
    val_dataset = CaptionDataset(val_df, images_dir, image_processor, tokenizer)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    return trainer, image_processor, tokenizer


def save_model(trainer: Trainer, image_processor: ViTImageProcessor, tokenizer, output_dir: Path) -> None:
    trainer.save_model(str(output_dir))
    image_processor.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
