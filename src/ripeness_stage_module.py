from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit


AUTOTUNE = tf.data.AUTOTUNE
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =========================================================
# 1. CONFIG
# =========================================================

@dataclass
class RipenessTrainConfig:
    data_dir: str
    output_dir: str = "ripeness_stage_runs"

    image_size: Tuple[int, int] = (260, 260)
    batch_size: int = 24
    seed: int = 42

    test_size: float = 0.15
    val_size: float = 0.15

    epochs_stage1: int = 8
    epochs_stage2: int = 10
    learning_rate_stage1: float = 1e-3
    learning_rate_stage2: float = 1e-5

    dense_units: int = 256
    dropout_rate: float = 0.30
    fine_tune_at: int = 260

    min_images_per_stage: int = 8
    min_stages_per_fruit: int = 3

    rejection_keep_correct_rate: float = 0.95
    model_version: str = "ripeness_stage_v2"


# =========================================================
# 2. UTILS
# =========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def inspect_image_size(image_path: str) -> Tuple[int, int]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    return int(image.shape[0]), int(image.shape[1])


# =========================================================
# 3. MANIFEST
# =========================================================

def build_ripeness_manifest(data_dir: str) -> pd.DataFrame:
    """
    Expected structure:
    data_dir/
        <routing_fruit>/
            <ripeness_stage>/
                *.jpg
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    records = []
    fruit_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda x: x.name)

    if not fruit_dirs:
        raise ValueError(f"No fruit folders found in: {root}")

    for fruit_dir in fruit_dirs:
        routing_fruit = fruit_dir.name
        stage_dirs = sorted([p for p in fruit_dir.iterdir() if p.is_dir()], key=lambda x: x.name)

        for stage_dir in stage_dirs:
            ripeness_stage = stage_dir.name

            for file_path in stage_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
                    records.append(
                        {
                            "filepath": str(file_path.resolve()),
                            "filename": file_path.name,
                            "routing_fruit": routing_fruit,
                            "ripeness_stage": ripeness_stage,
                        }
                    )

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise ValueError("No valid image files found in ripeness-stage dataset.")

    return manifest


def validate_manifest(
    manifest: pd.DataFrame,
    min_images_per_stage: int,
    min_stages_per_fruit: int,
) -> pd.DataFrame:
    counts = (
        manifest.groupby(["routing_fruit", "ripeness_stage"])
        .size()
        .reset_index(name="count")
    )

    valid_pairs = counts[counts["count"] >= min_images_per_stage].copy()
    if valid_pairs.empty:
        raise ValueError("No fruit-stage groups meet minimum sample threshold.")

    valid_stage_counts = (
        valid_pairs.groupby("routing_fruit")["ripeness_stage"]
        .nunique()
        .reset_index(name="num_stages")
    )

    valid_fruits = valid_stage_counts[
        valid_stage_counts["num_stages"] >= min_stages_per_fruit
    ]["routing_fruit"].tolist()

    filtered = manifest.merge(
        valid_pairs[["routing_fruit", "ripeness_stage"]],
        on=["routing_fruit", "ripeness_stage"],
        how="inner",
    )
    filtered = filtered[filtered["routing_fruit"].isin(valid_fruits)].reset_index(drop=True)

    if filtered.empty:
        raise ValueError("No trainable fruit groups remain after validation.")

    return filtered


def make_stage_label_mapping(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    stage_names = sorted(df["ripeness_stage"].unique().tolist())
    stage_to_idx = {name: i for i, name in enumerate(stage_names)}
    idx_to_stage = {i: name for name, i in stage_to_idx.items()}
    return stage_to_idx, idx_to_stage


def stratified_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = df["label"].values

    splitter_1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter_1.split(df, labels))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    val_relative = val_size / (1.0 - test_size)
    splitter_2 = StratifiedShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed)
    train_idx, val_idx = next(splitter_2.split(train_val_df, train_val_df["label"].values))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


# =========================================================
# 4. TF.DATA
# =========================================================

def decode_resize_image(path: tf.Tensor, label: tf.Tensor, image_size: Tuple[int, int]):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    return image, label


def build_dataset(
    df: pd.DataFrame,
    image_size: Tuple[int, int],
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (df["filepath"].astype(str).tolist(), df["label"].astype(np.int32).tolist())
    )

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, y: decode_resize_image(p, y, image_size),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# =========================================================
# 5. MODEL
# =========================================================

def build_ripeness_classifier(
    num_classes: int,
    image_size: Tuple[int, int],
    dense_units: int,
    dropout_rate: float,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3), name="image")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    backbone = tf.keras.applications.EfficientNetB2(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
        name="backbone",
    )
    backbone.trainable = False

    x = augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="embedding")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="ripeness_probs",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ripeness_stage_classifier")


def compile_classifier(model: tf.keras.Model, learning_rate: float, num_classes: int) -> None:
    k = min(2, num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="top1_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k, name=f"top{k}_acc"),
        ],
    )


def unfreeze_for_fine_tuning(model: tf.keras.Model, fine_tune_at: int) -> None:
    backbone = model.get_layer("backbone")
    backbone.trainable = True

    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False

    for layer in backbone.layers[fine_tune_at:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


# =========================================================
# 6. TRAIN / EVAL
# =========================================================

def get_callbacks(output_dir: Path) -> List[tf.keras.callbacks.Callback]:
    ensure_dir(output_dir)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_top1_acc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_top1_acc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ]


def collect_predictions(model: tf.keras.Model, ds: tf.data.Dataset):
    y_prob = model.predict(ds, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in ds], axis=0)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


def calibrate_rejection_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    keep_correct_rate: float,
) -> float:
    confidences = np.max(y_prob, axis=1)
    correct_mask = y_true == y_pred
    correct_conf = confidences[correct_mask]

    if len(correct_conf) == 0:
        return 0.50

    percentile = max(0.0, min(1.0, 1.0 - keep_correct_rate))
    threshold = float(np.quantile(correct_conf, percentile))
    return float(np.clip(threshold, 0.30, 0.95))


def train_single_fruit_model(
    routing_fruit: str,
    fruit_df: pd.DataFrame,
    cfg: RipenessTrainConfig,
    root_output_dir: Path,
) -> Dict:
    stage_to_idx, idx_to_stage = make_stage_label_mapping(fruit_df)
    fruit_df = fruit_df.copy()
    fruit_df["label"] = fruit_df["ripeness_stage"].map(stage_to_idx)

    train_df, val_df, test_df = stratified_split(
        fruit_df,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        seed=cfg.seed,
    )

    fruit_slug = slugify(routing_fruit)
    output_dir = root_output_dir / fruit_slug
    ensure_dir(output_dir)

    train_ds = build_dataset(train_df, cfg.image_size, cfg.batch_size, True, cfg.seed)
    val_ds = build_dataset(val_df, cfg.image_size, cfg.batch_size, False, cfg.seed)
    test_ds = build_dataset(test_df, cfg.image_size, cfg.batch_size, False, cfg.seed)

    model = build_ripeness_classifier(
        num_classes=len(stage_to_idx),
        image_size=cfg.image_size,
        dense_units=cfg.dense_units,
        dropout_rate=cfg.dropout_rate,
    )

    callbacks = get_callbacks(output_dir)

    compile_classifier(model, cfg.learning_rate_stage1, len(stage_to_idx))
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs_stage1,
        callbacks=callbacks,
        verbose=1,
    )

    unfreeze_for_fine_tuning(model, cfg.fine_tune_at)
    compile_classifier(model, cfg.learning_rate_stage2, len(stage_to_idx))
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs_stage1 + cfg.epochs_stage2,
        initial_epoch=cfg.epochs_stage1,
        callbacks=callbacks,
        verbose=1,
    )

    model = tf.keras.models.load_model(output_dir / "best_model.keras")

    val_true, val_pred, val_prob = collect_predictions(model, val_ds)
    rejection_threshold = calibrate_rejection_threshold(
        val_true,
        val_pred,
        val_prob,
        cfg.rejection_keep_correct_rate,
    )

    test_true, test_pred, test_prob = collect_predictions(model, test_ds)
    class_names = [idx_to_stage[i] for i in range(len(idx_to_stage))]

    report = classification_report(
        test_true,
        test_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv", index=True)

    top1 = float(np.mean(test_pred == test_true))
    topk = min(2, len(stage_to_idx))
    topk_acc = float(
        np.mean([true_label in np.argsort(prob_row)[-topk:] for true_label, prob_row in zip(test_true, test_prob)])
    )

    summary = {
        "routing_fruit": routing_fruit,
        "fruit_slug": fruit_slug,
        "model_version": cfg.model_version,
        "num_stages": len(stage_to_idx),
        "top1_accuracy": top1,
        f"top{topk}_accuracy": topk_acc,
        "rejection_threshold": rejection_threshold,
    }
    save_json(summary, output_dir / "metrics_summary.json")

    metadata = {
        "module_name": "ripeness_stage_classification",
        "routing_fruit": routing_fruit,
        "fruit_slug": fruit_slug,
        "model_version": cfg.model_version,
        "image_size": list(cfg.image_size),
        "num_stages": len(stage_to_idx),
        "stage_to_idx": stage_to_idx,
        "idx_to_stage": {str(v): k for k, v in stage_to_idx.items()},
        "rejection_threshold": rejection_threshold,
    }
    save_json(metadata, output_dir / "model_metadata.json")

    model.save(output_dir / "final_model.keras")

    fruit_df.to_csv(output_dir / "manifest_all.csv", index=False)
    train_df.to_csv(output_dir / "manifest_train.csv", index=False)
    val_df.to_csv(output_dir / "manifest_val.csv", index=False)
    test_df.to_csv(output_dir / "manifest_test.csv", index=False)

    return {
        "routing_fruit": routing_fruit,
        "fruit_slug": fruit_slug,
        "num_images": int(len(fruit_df)),
        "num_stages": int(len(stage_to_idx)),
        "stages": class_names,
        "model_dir": str(output_dir.resolve()),
    }


# =========================================================
# 7. INFERENCE CONTRACT
# =========================================================

@dataclass
class TopKRipenessPrediction:
    stage_name: str
    confidence: float


@dataclass
class RipenessStageOutput:
    module_name: str
    model_version: str
    routing_fruit: Optional[str]
    variety_name: Optional[str]
    quality_label: Optional[str]
    predicted_stage: Optional[str]
    predicted_index: Optional[int]
    confidence: float
    rejected: bool
    skipped: bool
    rejection_reason: Optional[str]
    skip_reason: Optional[str]
    top_k: List[TopKRipenessPrediction]
    probability_vector: List[float]
    embedding_vector: Optional[List[float]]
    original_image_size: Optional[Tuple[int, int]]
    model_input_size: Optional[Tuple[int, int]]


class RipenessStageService:
    def __init__(self, registry_path: str):
        with open(registry_path, "r", encoding="utf-8") as f:
            self.registry = json.load(f)

        self.root_dir = Path(registry_path).parent
        self.models: Dict[str, tf.keras.Model] = {}
        self.meta: Dict[str, Dict] = {}
        self.embedding_models: Dict[str, tf.keras.Model] = {}

        for routing_fruit, info in self.registry["fruit_models"].items():
            model_dir = self.root_dir / info["fruit_slug"]
            model = tf.keras.models.load_model(model_dir / "final_model.keras")

            with open(model_dir / "model_metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.models[routing_fruit] = model
            self.meta[routing_fruit] = metadata
            self.embedding_models[routing_fruit] = tf.keras.Model(
                inputs=model.input,
                outputs=model.get_layer("embedding").output,
            )

    def supported_fruits(self) -> List[str]:
        return sorted(list(self.models.keys()))

    def _load_single_image(self, image_path: str, image_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
        original_size = inspect_image_size(image_path)
        image_bytes = tf.io.read_file(image_path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR)
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, axis=0)
        return image.numpy(), original_size

    def predict(
        self,
        image_path: str,
        general_output: Dict,
        variety_output: Optional[Dict] = None,
        quality_output: Optional[Dict] = None,
        top_k: int = 3,
        return_embedding: bool = True,
    ) -> RipenessStageOutput:
        general_rejected = bool(general_output.get("rejected", False))
        routing_fruit = general_output.get("routing_fruit")

        variety_name = None if variety_output is None else variety_output.get("predicted_variety")
        quality_label = None if quality_output is None else quality_output.get("predicted_quality")

        if general_rejected or routing_fruit is None:
            return RipenessStageOutput(
                module_name="ripeness_stage_classification",
                model_version=self.registry["model_version"],
                routing_fruit=routing_fruit,
                variety_name=variety_name,
                quality_label=quality_label,
                predicted_stage=None,
                predicted_index=None,
                confidence=0.0,
                rejected=False,
                skipped=True,
                rejection_reason=None,
                skip_reason="general_classification_rejected_or_missing_routing",
                top_k=[],
                probability_vector=[],
                embedding_vector=None,
                original_image_size=None,
                model_input_size=None,
            )

        if routing_fruit not in self.models:
            return RipenessStageOutput(
                module_name="ripeness_stage_classification",
                model_version=self.registry["model_version"],
                routing_fruit=routing_fruit,
                variety_name=variety_name,
                quality_label=quality_label,
                predicted_stage=None,
                predicted_index=None,
                confidence=0.0,
                rejected=False,
                skipped=True,
                rejection_reason=None,
                skip_reason="unsupported_routing_fruit",
                top_k=[],
                probability_vector=[],
                embedding_vector=None,
                original_image_size=None,
                model_input_size=None,
            )

        model = self.models[routing_fruit]
        metadata = self.meta[routing_fruit]
        image_size = tuple(metadata["image_size"])

        x, original_size = self._load_single_image(image_path, image_size)
        probs = model.predict(x, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        idx_to_stage = {int(k): v for k, v in metadata["idx_to_stage"].items()}
        pred_stage = idx_to_stage[pred_idx]

        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_predictions = [
            TopKRipenessPrediction(
                stage_name=idx_to_stage[int(i)],
                confidence=float(probs[int(i)]),
            )
            for i in top_indices
        ]

        threshold = float(metadata["rejection_threshold"])
        rejected = confidence < threshold
        rejection_reason = "low_confidence" if rejected else None

        embedding_vector = None
        if return_embedding:
            embedding_vector = self.embedding_models[routing_fruit].predict(x, verbose=0)[0].astype(float).tolist()

        return RipenessStageOutput(
            module_name="ripeness_stage_classification",
            model_version=metadata["model_version"],
            routing_fruit=routing_fruit,
            variety_name=variety_name,
            quality_label=quality_label,
            predicted_stage=None if rejected else pred_stage,
            predicted_index=None if rejected else pred_idx,
            confidence=confidence,
            rejected=rejected,
            skipped=False,
            rejection_reason=rejection_reason,
            skip_reason=None,
            top_k=top_predictions,
            probability_vector=probs.astype(float).tolist(),
            embedding_vector=embedding_vector,
            original_image_size=original_size,
            model_input_size=image_size,
        )

    def to_pipeline_features(self, output: RipenessStageOutput) -> Dict[str, object]:
        return {
            "ripeness_routing_fruit": output.routing_fruit,
            "ripeness_variety_name": output.variety_name,
            "ripeness_quality_label": output.quality_label,
            "ripeness_predicted_stage": output.predicted_stage,
            "ripeness_predicted_index": output.predicted_index,
            "ripeness_confidence": output.confidence,
            "ripeness_rejected": output.rejected,
            "ripeness_skipped": output.skipped,
            "ripeness_probability_vector": output.probability_vector,
        }


# =========================================================
# 8. TRAINING RUNNER
# =========================================================

def run_training(cfg: RipenessTrainConfig) -> None:
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)

    manifest = build_ripeness_manifest(cfg.data_dir)
    manifest = validate_manifest(
        manifest,
        min_images_per_stage=cfg.min_images_per_stage,
        min_stages_per_fruit=cfg.min_stages_per_fruit,
    )

    registry = {
        "module_name": "ripeness_stage_classification",
        "model_version": cfg.model_version,
        "fruit_models": {},
    }

    fruit_summaries = []

    routing_fruits = sorted(manifest["routing_fruit"].unique().tolist())
    for routing_fruit in routing_fruits:
        fruit_df = manifest[manifest["routing_fruit"] == routing_fruit].reset_index(drop=True)
        result = train_single_fruit_model(routing_fruit, fruit_df, cfg, output_dir)
        fruit_summaries.append(result)

        registry["fruit_models"][routing_fruit] = {
            "fruit_slug": result["fruit_slug"],
            "num_stages": result["num_stages"],
            "stages": result["stages"],
        }

    pd.DataFrame(fruit_summaries).to_csv(output_dir / "training_summary.csv", index=False)
    save_json(registry, output_dir / "ripeness_registry.json")

    print(f"Ripeness-stage training complete. Registry: {output_dir / 'ripeness_registry.json'}")


if __name__ == "__main__":
    cfg = RipenessTrainConfig(
        data_dir=r"C:\Data\DataSetSmallV1\Ripeness Stage",
        output_dir="ripeness_stage_runs",
        image_size=(260, 260),
        batch_size=24,
        model_version="ripeness_stage_v2",
    )

    run_training(cfg)
