# Image Captioning

Generates natural-language captions for images. Originally a Kaggle notebook training an InceptionV3 + GloVe + LSTM encoder-decoder from scratch; rewritten as a runnable Python pipeline that fine-tunes a pretrained vision-language model.

## Pipeline

1. **Data loading** ([src/data_loader.py](src/data_loader.py)) - downloads the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset via `kagglehub`, or reads a local `captions.txt` + `Images/` directory. Splits captions into train/val/test by image, so all captions for a given photo stay in the same split.
2. **Dataset** ([src/dataset.py](src/dataset.py)) - a `torch.utils.data.Dataset` that loads each image, runs it through the model's image processor, and tokenizes its caption.
3. **Model fine-tuning** ([src/model.py](src/model.py)) - fine-tunes [`nlpconnect/vit-gpt2-image-captioning`](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) (a ViT image encoder + GPT-2 text decoder, already pretrained for captioning) via Hugging Face `Trainer`.
4. **Evaluation** ([src/evaluate.py](src/evaluate.py)) - generates a caption per test image (beam search) and scores it against Flickr8k's multiple human reference captions with BLEU.
5. **Visualization** ([src/visualize.py](src/visualize.py)) - saves a grid of sample test images with their generated captions as a PNG.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Kaggle API credentials are required for automatic dataset download - see [kagglehub's authentication docs](https://github.com/Kaggle/kagglehub#authenticate). Alternatively, download the Flickr8k dataset manually and pass its directory with `--data-dir`.

## Usage

```bash
# Full pipeline: download data, fine-tune, evaluate, save plots
python main.py --output-dir outputs --model-dir models/vit-gpt2-flickr8k

# Local data, capped image sample and fewer epochs for a quick run
python main.py --data-dir data/flickr8k --sample-size 200 --epochs 1
```

Outputs land in `outputs/` (`generated_captions.csv`, `sample_captions.png`) and the fine-tuned model in `--model-dir`.

## Tests

```bash
pytest
```

Fine-tuning `nlpconnect/vit-gpt2-image-captioning` instead starts from a model already trained on a large captioning corpus (COCO), with both its vision encoder and language decoder already competent at the task. Fine-tuning on Flickr8k then only needs to adapt that existing captioning ability to this dataset's images and caption style, rather than learning to caption from nothing - producing more fluent, accurate captions from the same amount of labeled data. Evaluation is also more rigorous: BLEU is computed against all of Flickr8k's reference captions per image, rather than only inspecting a couple of qualitative examples via greedy/beam search as the notebook did.
