"""Download and load the Flickr8k image captioning dataset from Kaggle."""

from pathlib import Path

import kagglehub
import pandas as pd

DATASET_SLUG = "adityajn105/flickr8k"
CAPTIONS_FILENAME = "captions.txt"
IMAGES_DIRNAME = "Images"


def download_dataset() -> Path:
    """Download the dataset via kagglehub and return the local directory path."""
    dataset_dir = kagglehub.dataset_download(DATASET_SLUG)
    return Path(dataset_dir)


def load_captions(data_dir: Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Load (image, caption) pairs and return (dataframe, images_dir).

    If data_dir is not provided, downloads the dataset via kagglehub first.
    """
    if data_dir is None:
        data_dir = download_dataset()

    captions_df = pd.read_csv(data_dir / CAPTIONS_FILENAME)
    captions_df = captions_df.dropna(subset=["image", "caption"])
    images_dir = data_dir / IMAGES_DIRNAME
    return captions_df, images_dir


def split_by_image(captions_df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 0):
    """Split captions into train/val/test by image (so all captions for an image stay together)."""
    image_ids = captions_df["image"].unique()
    rng = pd.Series(image_ids).sample(frac=1.0, random_state=seed)

    n_val = int(len(rng) * val_frac)
    n_test = int(len(rng) * test_frac)

    val_ids = set(rng.iloc[:n_val])
    test_ids = set(rng.iloc[n_val:n_val + n_test])

    is_val = captions_df["image"].isin(val_ids)
    is_test = captions_df["image"].isin(test_ids)

    train_df = captions_df[~is_val & ~is_test].reset_index(drop=True)
    val_df = captions_df[is_val].reset_index(drop=True)
    test_df = captions_df[is_test].reset_index(drop=True)
    return train_df, val_df, test_df
