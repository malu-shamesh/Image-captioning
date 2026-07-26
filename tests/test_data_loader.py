import pandas as pd

from src.data_loader import split_by_image


def _make_captions_df(n_images: int, captions_per_image: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n_images):
        image = f"img_{i}.jpg"
        for j in range(captions_per_image):
            rows.append({"image": image, "caption": f"caption {j} for {image}"})
    return pd.DataFrame(rows)


def test_split_by_image_keeps_all_captions_for_an_image_together():
    df = _make_captions_df(n_images=20, captions_per_image=3)
    train_df, val_df, test_df = split_by_image(df, val_frac=0.2, test_frac=0.2, seed=0)

    for split_df in (train_df, val_df, test_df):
        counts = split_df["image"].value_counts()
        assert (counts == 3).all()


def test_split_by_image_produces_disjoint_image_sets():
    df = _make_captions_df(n_images=20, captions_per_image=2)
    train_df, val_df, test_df = split_by_image(df, val_frac=0.2, test_frac=0.2, seed=0)

    train_images = set(train_df["image"])
    val_images = set(val_df["image"])
    test_images = set(test_df["image"])

    assert not (train_images & val_images)
    assert not (train_images & test_images)
    assert not (val_images & test_images)
    assert train_images | val_images | test_images == set(df["image"])
