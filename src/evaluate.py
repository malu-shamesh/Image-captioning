"""Generate captions for images and score them against reference captions with BLEU."""

from pathlib import Path

import evaluate as hf_evaluate
import pandas as pd
from PIL import Image
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

MAX_GENERATION_LENGTH = 32


def generate_caption(
    image_path: Path,
    model: VisionEncoderDecoderModel,
    image_processor: ViTImageProcessor,
    tokenizer: GPT2TokenizerFast,
    num_beams: int = 4,
) -> str:
    """Generate a caption for a single image."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    output_ids = model.generate(
        pixel_values, max_length=MAX_GENERATION_LENGTH, num_beams=num_beams
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def evaluate_bleu(
    test_df: pd.DataFrame,
    images_dir: Path,
    model: VisionEncoderDecoderModel,
    image_processor: ViTImageProcessor,
    tokenizer: GPT2TokenizerFast,
    output_dir: Path,
) -> dict:
    """Generate captions for each unique image in test_df and score against all its reference captions."""
    bleu = hf_evaluate.load("bleu")

    references_by_image = test_df.groupby("image")["caption"].apply(list).to_dict()

    predictions, references, rows = [], [], []
    for image_name, refs in references_by_image.items():
        caption = generate_caption(images_dir / image_name, model, image_processor, tokenizer)
        predictions.append(caption)
        references.append(refs)
        rows.append({"image": image_name, "generated_caption": caption, "reference_captions": refs})

    results = bleu.compute(predictions=predictions, references=references)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "generated_captions.csv", index=False)

    return results
