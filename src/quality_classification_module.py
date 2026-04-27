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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit


AUTOTUNE = tf.data.AUTOTUNE
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =========================================================
# 1. CONFIG
# =========================================================

@dataclass
class QualityTrainConfig:
    data_dir: str
    output_dir: str = "quality_classification_runs"

    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 24
    seed: int = 42

    test_size: float = 0.15
    val_size: float = 0.15

    epochs_stage1: int = 8
    epochs_stage2: int = 10
    learning_rate_stage1: float = 1e-3
    learning_rate_stage2: float = 1e-5

    dense_units: int = 192
    dropout_rate: float = 0.30
    fine_tune_at: int = 180

    min_images_per_quality_class: int = 10
    min_total_images_per_fruit: int = 40

    positive_label: str = "good"
    negative_label: str = "bad"

    rejection_keep_correct_rate: float = 0.95
    quality_decision_threshold: float = 0.50
    model_version: str = "quality_classification_v2"


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

def build_quality_manifest(data_dir: str) -> pd.DataFrame:
    """
    Expected structure:
    data_dir/
        <routing_fruit>/
            good/
            bad/
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
        quality_dirs = sorted([p for p in fruit_dir.iterdir() if p.is_dir()], key=lambda x: x.name)

        for quality_dir in quality_dirs:
            quality_label = quality_dir.name.strip().lower()

            for file_path in quality_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
                    records.append(
                        {
                            "filepath": str(file_path.resolve()),
                            "filename": file_path.name,
                            "routing_fruit": routing_fruit,
                            "quality_label": quality_label,
                        }
                    )

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise ValueError("No valid image files found in quality dataset.")

    return manifest


def validate_manifest(manifest: pd.DataFrame, cfg: QualityTrainConfig) -> pd.DataFrame:
    counts = (
        manifest.groupby(["routing_fruit", "quality_label"])
        .size()
        .reset_index(name="count")
    )

    valid_pairs = counts[counts["count"] >= cfg.min_images_per_quality_class].copy()
    if valid_pairs.empty:
        raise ValueError("No trainable fruit-quality groups remain after filtering.")

    filtered = manifest.merge(
        valid_pairs[["routing_fruit", "quality_label"]],
        on=["routing_fruit", "quality_label"],
        how="inner",
    )

    valid_fruits = []
    for routing_fruit, fruit_df in filtered.groupby("routing_fruit"):
        labels = set(fruit_df["quality_label"].str.lower().tolist())
        total_images = len(fruit_df)

        if (
            cfg.positive_label.lower() in labels
            and cfg.negative_label.lower() in labels
            and total_images >= cfg.min_total_images_per_fruit
        ):
            valid_fruits.append(routing_fruit)

    filtered = filtered[filtered["routing_fruit"].isin(valid_fruits)].reset_index(drop=True)

    if filtered.empty:
        raise ValueError("No valid fruit datasets remain after final validation.")

    return filtered


def encode_binary_labels(df: pd.DataFrame, positive_label: str) -> pd.DataFrame:
    df = df.copy()
    df["label"] = (df["quality_label"].str.lower() == positive_label.lower()).astype(np.int32)
    return df


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
        (
            df["filepath"].astype(str).tolist(),
            df["label"].astype(np.float32).tolist(),
        )
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

def build_quality_classifier(
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

    backbone = tf.keras.applications.MobileNetV3Large(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
        name="backbone",
    )
    backbone.trainable = False

    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="embedding")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="quality_prob")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="quality_classifier")


def compile_classifier(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
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
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
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
    y_prob = model.predict(ds, verbose=0).reshape(-1)
    y_true = np.concatenate([y.numpy() for _, y in ds], axis=0).astype(np.int32)
    return y_true, y_prob


def calibrate_rejection_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    decision_threshold: float,
    keep_correct_rate: float,
) -> float:
    y_pred = (y_prob >= decision_threshold).astype(np.int32)
    correct_mask = y_pred == y_true

    if correct_mask.sum() == 0:
        return 0.05

    certainty = np.abs(y_prob - 0.5) * 2.0
    correct_certainty = certainty[correct_mask]

    percentile = max(0.0, min(1.0, 1.0 - keep_correct_rate))
    threshold = float(np.quantile(correct_certainty, percentile))
    return float(np.clip(threshold, 0.02, 0.95))


def train_single_fruit_model(
    routing_fruit: str,
    fruit_df: pd.DataFrame,
    cfg: QualityTrainConfig,
    root_output_dir: Path,
) -> Dict:
    fruit_df = encode_binary_labels(fruit_df, cfg.positive_label)

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

    model = build_quality_classifier(
        image_size=cfg.image_size,
        dense_units=cfg.dense_units,
        dropout_rate=cfg.dropout_rate,
    )

    callbacks = get_callbacks(output_dir)

    compile_classifier(model, cfg.learning_rate_stage1)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs_stage1,
        callbacks=callbacks,
        verbose=1,
    )

    unfreeze_for_fine_tuning(model, cfg.fine_tune_at)
    compile_classifier(model, cfg.learning_rate_stage2)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs_stage1 + cfg.epochs_stage2,
        initial_epoch=cfg.epochs_stage1,
        callbacks=callbacks,
        verbose=1,
    )

    model = tf.keras.models.load_model(output_dir / "best_model.keras")

    val_true, val_prob = collect_predictions(model, val_ds)
    rejection_threshold = calibrate_rejection_threshold(
        y_true=val_true,
        y_prob=val_prob,
        decision_threshold=cfg.quality_decision_threshold,
        keep_correct_rate=cfg.rejection_keep_correct_rate,
    )

    test_true, test_prob = collect_predictions(model, test_ds)
    test_pred = (test_prob >= cfg.quality_decision_threshold).astype(np.int32)

    acc = float(accuracy_score(test_true, test_pred))
    prec = float(precision_score(test_true, test_pred, zero_division=0))
    rec = float(recall_score(test_true, test_pred, zero_division=0))
    f1 = float(f1_score(test_true, test_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(test_true, test_prob))
    except ValueError:
        auc = float("nan")

    class_report = classification_report(
        test_true,
        test_pred,
        target_names=[cfg.negative_label, cfg.positive_label],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(class_report).transpose().to_csv(output_dir / "classification_report.csv", index=True)

    certainty = np.abs(test_prob - 0.5) * 2.0
    accepted_mask = certainty >= rejection_threshold
    rejection_rate = float(1.0 - accepted_mask.mean())

    metrics_summary = {
        "routing_fruit": routing_fruit,
        "fruit_slug": fruit_slug,
        "model_version": cfg.model_version,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc": auc,
        "decision_threshold": cfg.quality_decision_threshold,
        "rejection_threshold": rejection_threshold,
        "test_rejection_rate": rejection_rate,
    }
    save_json(metrics_summary, output_dir / "metrics_summary.json")

    model.save(output_dir / "final_model.keras")

    metadata = {
        "module_name": "quality_classification",
        "routing_fruit": routing_fruit,
        "fruit_slug": fruit_slug,
        "model_version": cfg.model_version,
        "image_size": list(cfg.image_size),
        "decision_threshold": cfg.quality_decision_threshold,
        "rejection_threshold": rejection_threshold,
        "positive_label": cfg.positive_label,
        "negative_label": cfg.negative_label,
    }
    save_json(metadata, output_dir / "model_metadata.json")

    fruit_df.to_csv(output_dir / "manifest_all.csv", index=False)
    train_df.to_csv(output_dir / "manifest_train.csv", index=False)
    val_df.to_csv(output_dir / "manifest_val.csv", index=False)
    test_df.to_csv(output_dir / "manifest_test.csv", index=False)

    return {
        "routing_fruit": routing_fruit,
        "fruit_slug": fruit_slug,
        "num_images": int(len(fruit_df)),
        "num_good": int((fruit_df["label"] == 1).sum()),
        "num_bad": int((fruit_df["label"] == 0).sum()),
        "model_dir": str(output_dir.resolve()),
    }


# =========================================================
# 7. INFERENCE CONTRACT
# =========================================================

@dataclass
class QualityClassificationOutput:
    module_name: str
    model_version: str
    routing_fruit: Optional[str]
    variety_name: Optional[str]
    predicted_quality: Optional[str]
    quality_score_good: float
    quality_score_bad: float
    confidence: float
    rejected: bool
    skipped: bool
    rejection_reason: Optional[str]
    skip_reason: Optional[str]
    decision_threshold: Optional[float]
    rejection_threshold: Optional[float]
    embedding_vector: Optional[List[float]]
    original_image_size: Optional[Tuple[int, int]]
    model_input_size: Optional[Tuple[int, int]]


class QualityClassifierService:
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
        return_embedding: bool = True,
    ) -> QualityClassificationOutput:
        general_rejected = bool(general_output.get("rejected", False))
        routing_fruit = general_output.get("routing_fruit")
        variety_name = None if variety_output is None else variety_output.get("predicted_variety")

        if general_rejected or routing_fruit is None:
            return QualityClassificationOutput(
                module_name="quality_classification",
                model_version=self.registry["model_version"],
                routing_fruit=routing_fruit,
                variety_name=variety_name,
                predicted_quality=None,
                quality_score_good=0.0,
                quality_score_bad=0.0,
                confidence=0.0,
                rejected=False,
                skipped=True,
                rejection_reason=None,
                skip_reason="general_classification_rejected_or_missing_routing",
                decision_threshold=None,
                rejection_threshold=None,
                embedding_vector=None,
                original_image_size=None,
                model_input_size=None,
            )

        if routing_fruit not in self.models:
            return QualityClassificationOutput(
                module_name="quality_classification",
                model_version=self.registry["model_version"],
                routing_fruit=routing_fruit,
                variety_name=variety_name,
                predicted_quality=None,
                quality_score_good=0.0,
                quality_score_bad=0.0,
                confidence=0.0,
                rejected=False,
                skipped=True,
                rejection_reason=None,
                skip_reason="unsupported_routing_fruit",
                decision_threshold=None,
                rejection_threshold=None,
                embedding_vector=None,
                original_image_size=None,
                model_input_size=None,
            )

        model = self.models[routing_fruit]
        metadata = self.meta[routing_fruit]
        image_size = tuple(metadata["image_size"])

        x, original_size = self._load_single_image(image_path, image_size)
        prob_good = float(model.predict(x, verbose=0).reshape(-1)[0])
        prob_bad = float(1.0 - prob_good)

        decision_threshold = float(metadata["decision_threshold"])
        rejection_threshold = float(metadata["rejection_threshold"])

        certainty = float(abs(prob_good - 0.5) * 2.0)
        rejected = certainty < rejection_threshold
        rejection_reason = "low_confidence" if rejected else None

        predicted_quality = metadata["positive_label"] if prob_good >= decision_threshold else metadata["negative_label"]
        if rejected:
            predicted_quality = None

        confidence = float(max(prob_good, prob_bad))

        embedding_vector = None
        if return_embedding:
            embedding_vector = self.embedding_models[routing_fruit].predict(x, verbose=0)[0].astype(float).tolist()

        return QualityClassificationOutput(
            module_name="quality_classification",
            model_version=metadata["model_version"],
            routing_fruit=routing_fruit,
            variety_name=variety_name,
            predicted_quality=predicted_quality,
            quality_score_good=prob_good,
            quality_score_bad=prob_bad,
            confidence=confidence,
            rejected=rejected,
            skipped=False,
            rejection_reason=rejection_reason,
            skip_reason=None,
            decision_threshold=decision_threshold,
            rejection_threshold=rejection_threshold,
            embedding_vector=embedding_vector,
            original_image_size=original_size,
            model_input_size=image_size,
        )

    def to_pipeline_features(self, output: QualityClassificationOutput) -> Dict[str, object]:
        return {
            "quality_routing_fruit": output.routing_fruit,
            "quality_variety_name": output.variety_name,
            "quality_predicted_class": output.predicted_quality,
            "quality_confidence": output.confidence,
            "quality_rejected": output.rejected,
            "quality_skipped": output.skipped,
            "quality_score_good": output.quality_score_good,
            "quality_score_bad": output.quality_score_bad,
        }


# =========================================================
# 8. TRAINING RUNNER
# =========================================================

def run_training(cfg: QualityTrainConfig) -> None:
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)

    manifest = build_quality_manifest(cfg.data_dir)
    manifest = validate_manifest(manifest, cfg)

    registry = {
        "module_name": "quality_classification",
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
            "num_images": result["num_images"],
        }

    pd.DataFrame(fruit_summaries).to_csv(output_dir / "training_summary.csv", index=False)
    save_json(registry, output_dir / "quality_registry.json")

    print(f"Quality training complete. Registry: {output_dir / 'quality_registry.json'}")


if __name__ == "__main__":
    cfg = QualityTrainConfig(
        data_dir=r"C:\Data\DataSetSmallV1\Quality control",
        output_dir="quality_classification_runs",
        image_size=(224, 224),
        batch_size=24,
        positive_label="good",
        negative_label="bad",
        model_version="quality_classification_v2",
    )

    run_training(cfg)
