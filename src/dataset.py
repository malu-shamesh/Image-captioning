"""torch Dataset wrapping (image, caption) pairs for the vision-encoder-decoder model."""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

MAX_CAPTION_LENGTH = 64


class CaptionDataset(Dataset):
    """Yields pixel_values (from the image processor) and labels (tokenized caption)."""

    def __init__(self, captions_df: pd.DataFrame, images_dir: Path, image_processor, tokenizer):
        self.captions_df = captions_df.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.captions_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.captions_df.iloc[idx]
        image = Image.open(self.images_dir / row["image"]).convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values[0]

        labels = self.tokenizer(
            row["caption"],
            padding="max_length",
            max_length=MAX_CAPTION_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}
