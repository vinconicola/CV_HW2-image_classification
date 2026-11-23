# Oxford-IIIT Pet image classification

This project compares three transfer-learning strategies for classifying the 37 breeds in the Oxford-IIIT Pet dataset: full MobileNetV3-Small fine-tuning, classifier-head training with a frozen feature extractor, and frozen pooled features classified by a scikit-learn linear SVM. It provides a reproducible pipeline for training, validation-based model selection, and held-out test evaluation.

## Dataset and license

The [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) contains 37 breed categories. Its official page states that it is available for research and commercial use under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**. The images are not redistributed here. Run with `--download` to let TorchVision download them, or follow the official page and place the extracted dataset under `data/oxford-iiit-pet/`.

TorchVision loads the official `trainval` and `test` partitions with breed (`category`) targets. A fixed-seed, stratified 80/20 split of `trainval` creates training and validation sets. The exact indices and per-breed counts are saved in `results/splits.json`. The official test set is used only for final evaluation after model/checkpoint selection.

## Environment

Use Python 3.11 or 3.12. Install PyTorch and TorchVision using the [official installer](https://pytorch.org/get-started/locally/), then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

The run writes exact Python, PyTorch, TorchVision, scikit-learn and NumPy versions to `results/environment.json`. CUDA availability is detected automatically. Seeds are set to 42 for Python, NumPy, and PyTorch; deterministic cuDNN and PyTorch algorithms are requested (`warn_only=True`), so operations without deterministic implementations and hardware/library variation may still vary.

## Reproduce

Run all approaches (default: 15 epochs maximum, batch size 32):

```bash
python main.py --download --data data --output results
```

Run individual approaches with the same partitions:

```bash
python main.py --approach full --data data --output results
python main.py --approach head --data data --output results
python main.py --approach features --data data --output results
```

The first execution downloads the official data and the ImageNet pretrained weights. The neural approaches use cross-entropy and AdamW (`1e-4` full fine-tuning; `1e-3` classifier head; weight decay `1e-4`). Early stopping uses validation macro-F1, patience 3, and saves the best validation checkpoint. The head approach trains every parameter in `model.classifier` (the 576→1024 linear layer, its activation/dropout, and the replacement 1024→37 linear layer); `model.features`, including its BatchNorm state, is frozen. Full fine-tuning trains all parameters. Both use the same augmentation (random resized crop to 224, horizontal flip) only on training images. Evaluation uses the pretrained weights' resize, center crop, and ImageNet mean/std normalization.

The feature approach extracts the 576-dimensional output after `features` and adaptive average pooling. It compares linear SVM `C` values 0.1, 1, and 10 on validation macro-F1. A `StandardScaler` is fitted on training features only. It does not optimize a neural loss. Model selection never uses test data.

`results/` contains split counts and indices, selected checkpoints, per-approach metrics, confusion matrices, predictions, a summary CSV/JSON, and environment versions. The predictions CSV can be used to inspect correct and incorrect examples by class; these are image-level breed predictions, and source image files remain in the dataset directory.


## Files

- `main.py`: data loading, training, validation-based selection, feature extraction, and evaluation.
- `requirements.txt`: scikit-learn dependencies (install torch/torchvision separately).
- `results/`: generated artifacts; checkpoints are reproducible outputs and are not committed by default.
