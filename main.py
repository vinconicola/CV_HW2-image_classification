"""Reproducible Oxford-IIIT Pet transfer-learning experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

SEED = 42
NUM_CLASSES = 37
VAL_FRACTION = 0.20
BATCH_SIZE = 32
EPOCHS = 15
PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Some CUDA operations may not have deterministic implementations.
    torch.use_deterministic_algorithms(True, warn_only=True)


class TransformSubset(Dataset):
    """Subset selecting one transform while retaining torchvision's labels."""
    def __init__(self, base: Dataset, indices: list[int], transform):
        self.base, self.indices, self.transform = base, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, label = self.base[self.indices[i]]
        return self.transform(image), label


def make_transforms(weights):
    # Same normalization and evaluation resize/crop prescribed by the weights.
    preset = weights.transforms()
    mean, std = preset.mean, preset.std
    eval_transform = transforms.Compose([
        transforms.Resize(preset.resize_size, interpolation=preset.interpolation),
        transforms.CenterCrop(preset.crop_size),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(preset.crop_size, scale=(0.75, 1.0),
                                    interpolation=preset.interpolation),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, eval_transform


def class_count_table(labels: np.ndarray, indices: list[int], names: list[str]):
    counts = np.bincount(labels[indices], minlength=len(names))
    return {name: int(count) for name, count in zip(names, counts)}


def data_bundle(root: Path, out: Path, download: bool, weights):
    base_train = datasets.OxfordIIITPet(root=str(root), split="trainval",
                                         target_types="category", download=download)
    base_test = datasets.OxfordIIITPet(root=str(root), split="test",
                                        target_types="category", download=download)
    labels = np.asarray(base_train._labels, dtype=np.int64)
    names = list(base_train.classes)
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)).tolist(), test_size=VAL_FRACTION, random_state=SEED,
        stratify=labels)
    train_idx, val_idx = sorted(train_idx), sorted(val_idx)
    out.mkdir(parents=True, exist_ok=True)
    split_data = {"seed": SEED, "validation_fraction": VAL_FRACTION,
                  "train_indices": train_idx, "validation_indices": val_idx,
                  "class_names": names,
                  "counts": {"train": class_count_table(labels, train_idx, names),
                             "validation": class_count_table(labels, val_idx, names),
                             "test": class_count_table(np.asarray(base_test._labels),
                                                       list(range(len(base_test))), names)}}
    split_file = out / "splits.json"
    if split_file.exists():
        stored = json.loads(split_file.read_text())
        if stored["seed"] != SEED or stored["train_indices"] != train_idx or stored["validation_indices"] != val_idx:
            raise RuntimeError("Existing split manifest does not match current seed/dataset")
    else:
        split_file.write_text(json.dumps(split_data, indent=2))
    train_tf, eval_tf = make_transforms(weights)
    train_ds = TransformSubset(base_train, train_idx, train_tf)
    train_eval_ds = TransformSubset(base_train, train_idx, eval_tf)
    val_ds = TransformSubset(base_train, val_idx, eval_tf)
    test_ds = TransformSubset(base_test, list(range(len(base_test))), eval_tf)
    return names, train_ds, train_eval_ds, val_ds, test_ds, split_data


def build_model(weights, full_finetune: bool):
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    if not full_finetune:
        for p in model.features.parameters():
            p.requires_grad = False
    return model.to(DEVICE)


def loader(ds, shuffle=False):
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=2, pin_memory=DEVICE.type == "cuda",
                      generator=generator if shuffle else None)


@torch.no_grad()
def predict(model, data_loader):
    model.eval()
    y_true, y_pred = [], []
    for x, y in data_loader:
        logits = model(x.to(DEVICE, non_blocking=True))
        y_true.extend(y.numpy().tolist())
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def score(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def save_evaluation(tag, y, pred, names, out, image_ids=None):
    metrics = score(y, pred)
    metrics["confusion_matrix"] = confusion_matrix(y, pred, labels=range(len(names))).tolist()
    (out / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(metrics["confusion_matrix"], cmap="Blues", interpolation="nearest")
    ticks = np.arange(len(names))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("Predicted breed"); ax.set_ylabel("True breed")
    ax.set_title(f"{tag}: confusion matrix (test set)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_confusion.png", dpi=180)
    plt.close(fig)
    with (out / f"{tag}_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id", "true_index", "true_breed", "pred_index", "pred_breed", "correct"])
        image_ids = image_ids or [""] * len(y)
        w.writerows((image_id, int(a), names[int(a)], int(b), names[int(b)], bool(a == b))
                    for image_id, a, b in zip(image_ids, y, pred))
    return metrics


def fit_neural(tag, train_ds, train_eval_ds, val_ds, test_ds, names, out,
               test_image_ids, full_finetune: bool, epochs: int, lr: float):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = build_model(weights, full_finetune)
    trainable = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    train_loader = loader(train_ds, shuffle=True)
    best_f1, best_epoch, stale = -1.0, 0, 0
    checkpoint = out / f"{tag}_best.pt"
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        if not full_finetune:
            # Freeze convolutional weights and their BatchNorm running statistics.
            model.features.eval()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
            running += loss.item() * len(y); seen += len(y)
        vy, vp = predict(model, loader(val_ds))
        val_score = score(vy, vp)
        print(f"{tag} epoch {epoch+1}/{epochs}: loss={running/seen:.4f} val_f1={val_score['macro_f1']:.4f}")
        if val_score["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = val_score["macro_f1"], epoch + 1, 0
            torch.save({"model": model.state_dict(), "epoch": best_epoch,
                        "val_macro_f1": best_f1, "seed": SEED}, checkpoint)
        else:
            stale += 1
            if stale >= PATIENCE: break
    elapsed = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True)["model"])
    # Test is evaluated once after the validation-selected checkpoint is fixed.
    ty, tp = predict(model, loader(test_ds))
    result = save_evaluation(tag, ty, tp, names, out, test_image_ids)
    try:
        try_, trp = predict(model, loader(train_eval_ds))
        result["train_accuracy"] = float(accuracy_score(try_, trp))
    except Exception as e:
        print(f"Could not compute train score: {e}")
    result.update({"approach": tag, "trainable_parameters": sum(p.numel() for p in trainable),
                   "training_seconds": elapsed, "selected_epoch": best_epoch,
                   "selected_validation_macro_f1": best_f1, "learning_rate": lr,
                   "batch_size": BATCH_SIZE, "epochs_max": epochs,
                   "optimizer": "AdamW", "loss": "CrossEntropyLoss"})
    (out / f"{tag}_metrics.json").write_text(json.dumps(result, indent=2))
    return result


@torch.no_grad()
def extract_features(model, ds):
    model.eval()
    feats, labels = [], []
    for x, y in loader(ds):
        x = x.to(DEVICE, non_blocking=True)
        z = model.avgpool(model.features(x)).flatten(1)
        feats.append(z.cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def fit_svm(train_ds, val_ds, test_ds, names, out, test_image_ids):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights).to(DEVICE)
    for p in model.parameters(): p.requires_grad = False
    start = time.perf_counter()
    xtr, ytr = extract_features(model, train_ds)
    xv, yv = extract_features(model, val_ds)
    extract_time = time.perf_counter() - start
    best, best_c = -1, None
    best_pipe = None
    clf_start = time.perf_counter()
    for c in (0.1, 1.0, 10.0):
        pipe = make_pipeline(StandardScaler(), SVC(C=c, kernel="linear", class_weight=None))
        pipe.fit(xtr, ytr)
        vf1 = f1_score(yv, pipe.predict(xv), average="macro", zero_division=0)
        if vf1 > best:
            best, best_c, best_pipe = vf1, c, pipe
    clf_time = time.perf_counter() - clf_start
    # Extract test features only after validation has selected the classifier.
    test_start = time.perf_counter()
    xt, yt = extract_features(model, test_ds)
    extract_time += time.perf_counter() - test_start
    pred = best_pipe.predict(xt)
    result = save_evaluation("feature_svm", yt, pred, names, out, test_image_ids)
    result.update({"approach": "feature_svm", "trainable_parameters": 0,
                   "training_seconds": extract_time + clf_time,
                   "feature_extraction_seconds": extract_time,
                   "classifier_training_seconds": clf_time,
                   "selected_C": best_c, "selected_validation_macro_f1": best,
                   "classifier": "StandardScaler + linear SVM", "feature_dimension": int(xtr.shape[1]),
                   "scaler_fit_on": "training features only"})
    (out / "feature_svm_metrics.json").write_text(json.dumps(result, indent=2))
    return result


def write_summary(results, out):
    (out / "results.json").write_text(json.dumps(results, indent=2))
    fields = ["approach", "accuracy", "macro_f1", "train_accuracy", "trainable_parameters",
              "training_seconds", "selected_epoch", "selected_C"]
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(results)


def main():
    global BATCH_SIZE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--download", action="store_true", help="download dataset through torchvision")
    p.add_argument("--approach", choices=["all", "full", "head", "features"], default="all")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr-full", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    args = p.parse_args()
    BATCH_SIZE = args.batch_size
    seed_everything()
    args.output.mkdir(parents=True, exist_ok=True)
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    names, train_ds, train_eval_ds, val_ds, test_ds, splits = data_bundle(args.data, args.output, args.download, weights)
    # TorchVision's official test annotation order is authoritative, so retain that order.
    test_base = datasets.OxfordIIITPet(root=str(args.data), split="test", target_types="category")
    test_image_ids = [path.name for path in test_base._images]
    results = []
    if args.approach in ("all", "full"):
        results.append(fit_neural("full_finetune", train_ds, train_eval_ds, val_ds, test_ds, names,
                                  args.output, test_image_ids, True, args.epochs, args.lr_full))
    if args.approach in ("all", "head"):
        results.append(fit_neural("classifier_head", train_ds, train_eval_ds, val_ds, test_ds, names,
                                  args.output, test_image_ids, False, args.epochs, args.lr_head))
    if args.approach in ("all", "features"):
        results.append(fit_svm(train_eval_ds, val_ds, test_ds, names, args.output, test_image_ids))
    write_summary(results, args.output)
    (args.output / "environment.json").write_text(json.dumps({
        "python": __import__("sys").version, "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__, "scikit_learn": sklearn.__version__,
        "numpy": np.__version__, "seed": SEED, "deterministic_algorithms": True,
        "device": str(DEVICE), "weights": "MobileNet_V3_Small_Weights.DEFAULT"
    }, indent=2))
    print(f"Finished. Results are in {args.output.resolve()}")


if __name__ == "__main__":
    main()
