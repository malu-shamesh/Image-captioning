"""CLI entrypoint for the image captioning pipeline.

Usage:
    python main.py --output-dir outputs --model-dir models/vit-gpt2-flickr8k
    python main.py --data-dir data/flickr8k --sample-size 200 --epochs 1  # quick smoke test
"""

import argparse
import pandas as pd
from pathlib import Path

from src.data_loader import load_captions, split_by_image
from src.evaluate import evaluate_bleu
from src.model import save_model, train_model
from src.visualize import plot_sample_captions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image captioning fine-tuning pipeline")
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory containing captions.txt and an Images/ folder. If omitted, downloads via kagglehub.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"),
        help="Directory to write plots and evaluation reports to.",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models/vit-gpt2-flickr8k"),
        help="Directory to save the fine-tuned model to.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs.")
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Optional cap on number of unique images used, for a fast smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading captions...")
    captions_df, images_dir = load_captions(args.data_dir)

    if args.sample_size:
        sampled_images = captions_df["image"].unique()[: args.sample_size]
        captions_df = captions_df[captions_df["image"].isin(sampled_images)]

    train_df, val_df, test_df = split_by_image(captions_df)
    print(f"Train captions: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    print(f"Fine-tuning model for {args.epochs} epoch(s)...")
    trainer, image_processor, tokenizer = train_model(
        train_df, val_df, images_dir, output_dir=args.model_dir, epochs=args.epochs
    )
    save_model(trainer, image_processor, tokenizer, args.model_dir)
    print(f"Saved model to {args.model_dir}")

    print("Generating captions and scoring BLEU on the held-out test set...")
    bleu_results = evaluate_bleu(
        test_df, images_dir, trainer.model, image_processor, tokenizer, args.output_dir
    )
    print(f"BLEU: {bleu_results['bleu']:.3f}")

    generated_df = pd.read_csv(args.output_dir / "generated_captions.csv")
    plot_sample_captions(generated_df, images_dir, args.output_dir)
    print(f"Saved evaluation report and sample captions to {args.output_dir}")


if __name__ == "__main__":
    main()
