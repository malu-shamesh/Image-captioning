"""Save a grid of sample images alongside their generated captions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def plot_sample_captions(
    generated_captions_df: pd.DataFrame, images_dir: Path, output_dir: Path, n_samples: int = 6
) -> None:
    sample = generated_captions_df.sample(n=min(n_samples, len(generated_captions_df)), random_state=0)

    n_cols = 3
    n_rows = (len(sample) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, sample.iterrows()):
        image = Image.open(images_dir / row["image"])
        ax.imshow(image)
        ax.set_title(row["generated_caption"], fontsize=10, wrap=True)
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "sample_captions.png", bbox_inches="tight")
    plt.close(fig)
