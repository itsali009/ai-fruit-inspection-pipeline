from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
AUTOTUNE = tf.data.AUTOTUNE


# =========================================================
# 1. CONFIG
# =========================================================

@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str = "general_classification_runs"

    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    seed: int = 42

    test_size: float = 0.15
    val_size: float = 0.15

    epochs_stage1: int = 8
    epochs_stage2: int = 12
    learning_rate_stage1: float = 1e-3
    learning_rate_stage2: float = 1e-5

    dense_units: int = 256
    dropout_rate: float = 0.30
    fine_tune_at: int = 140

    min_images_per_class: int = 5
    rejection_keep_correct_rate: float = 0.95
    model_version: str = "general_classification_v2"

    # maps exact class -> routing fruit
    # example: {"Banana Namwa": "Banana", "Banana Plantain": "Banana"}
    class_to_routing_fruit: Optional[Dict[str, str]] = None


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


def inspect_image_size(image_path: str) -> Tuple[int, int]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    shape = image.shape
    return int(shape[0]), int(shape[1])


def is_expected_image_size(image_path: str, expected_size: Tuple[int, int]) -> bool:
    h, w = inspect_image_size(image_path)
    return (h, w) == expected_size


# =========================================================
# 3. MANIFEST
# =========================================================

def build_manifest(
    data_dir: str,
    class_to_routing_fruit: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    records = []
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)

    if not class_dirs:
        raise ValueError(f"No class folders found in: {root}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        routing_fruit = (
            class_to_routing_fruit[class_name]
            if class_to_routing_fruit and class_name in class_to_routing_fruit
            else class_name
        )

        for file_path in class_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
                records.append(
                    {
                        "filepath": str(file_path.resolve()),
                        "filename": file_path.name,
                        "class_name": class_name,
                        "routing_fruit": routing_fruit,
                    }
                )

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise ValueError("No valid image files found.")

    return manifest


def validate_manifest(manifest: pd.DataFrame, min_images_per_class: int = 5) -> None:
    if "class_name" not in manifest.columns:
        raise ValueError("Manifest must contain 'class_name'.")

    counts = manifest["class_name"].value_counts().sort_index()
    if counts.shape[0] < 2:
        raise ValueError("At least 2 classes are required.")

    too_small = counts[counts < min_images_per_class]
    if not too_small.empty:
        raise ValueError(
            "Some classes have too few images:\n"
            + too_small.to_string()
            + f"\nIncrease samples or lower min_images_per_class."
        )


def make_label_mapping(manifest: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    class_names = sorted(manifest["class_name"].unique().tolist())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


def stratified_split(
    manifest: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = manifest["label"].values

    splitter_1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter_1.split(manifest, labels))

    train_val_df = manifest.iloc[train_val_idx].reset_index(drop=True)
    test_df = manifest.iloc[test_idx].reset_index(drop=True)

    val_relative_size = val_size / (1.0 - test_size)
    splitter_2 = StratifiedShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=seed)
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
    paths = df["filepath"].astype(str).tolist()
    labels = df["label"].astype(np.int32).tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

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

def build_general_classifier(
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

    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        name="backbone",
    )
    backbone.trainable = False

    x = augmentation(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="embedding")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="class_probs",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="general_fruit_classifier")


def compile_classifier(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="top1_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
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
# 6. TRAINING
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


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    cfg: TrainConfig,
    output_dir: Path,
) -> tf.keras.Model:
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

    return tf.keras.models.load_model(output_dir / "best_model.keras")


# =========================================================
# 7. EVALUATION
# =========================================================

def collect_predictions(model: tf.keras.Model, ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_prob = model.predict(ds, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in ds], axis=0)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


def calibrate_rejection_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    keep_correct_rate: float = 0.95,
) -> float:
    confidences = np.max(y_prob, axis=1)
    correct_mask = y_true == y_pred
    correct_confidences = confidences[correct_mask]

    if len(correct_confidences) == 0:
        return 0.50

    percentile = max(0.0, min(1.0, 1.0 - keep_correct_rate))
    threshold = float(np.quantile(correct_confidences, percentile))
    return float(np.clip(threshold, 0.30, 0.95))


def evaluate_and_save_reports(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    idx_to_class: Dict[int, str],
    output_dir: Path,
    cfg: TrainConfig,
) -> float:
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    val_true, val_pred, val_prob = collect_predictions(model, val_ds)
    rejection_threshold = calibrate_rejection_threshold(
        val_true,
        val_pred,
        val_prob,
        keep_correct_rate=cfg.rejection_keep_correct_rate,
    )

    test_true, test_pred, test_prob = collect_predictions(model, test_ds)

    report = classification_report(
        test_true,
        test_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv", index=True)

    top1 = float(np.mean(test_pred == test_true))
    top3 = float(
        np.mean(
            [true_label in np.argsort(prob_row)[-3:] for true_label, prob_row in zip(test_true, test_prob)]
        )
    )

    accepted_mask = np.max(test_prob, axis=1) >= rejection_threshold
    rejection_rate = float(1.0 - accepted_mask.mean())

    save_json(
        {
            "model_version": cfg.model_version,
            "num_classes": len(class_names),
            "top1_accuracy": top1,
            "top3_accuracy": top3,
            "rejection_threshold": rejection_threshold,
            "test_rejection_rate": rejection_rate,
        },
        output_dir / "metrics_summary.json",
    )

    return rejection_threshold


# =========================================================
# 8. EXPORT
# =========================================================

def export_artifacts(
    model: tf.keras.Model,
    class_to_idx: Dict[str, int],
    class_to_routing_fruit: Dict[str, str],
    rejection_threshold: float,
    cfg: TrainConfig,
    manifest: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    model.save(output_dir / "final_model.keras")

    metadata = {
        "module_name": "general_classification",
        "model_version": cfg.model_version,
        "image_size": list(cfg.image_size),
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
        "class_to_routing_fruit": class_to_routing_fruit,
        "supported_routing_fruits": sorted(set(class_to_routing_fruit.values())),
        "rejection_threshold": rejection_threshold,
    }
    save_json(metadata, output_dir / "model_metadata.json")

    manifest.to_csv(output_dir / "manifest_all.csv", index=False)
    train_df.to_csv(output_dir / "manifest_train.csv", index=False)
    val_df.to_csv(output_dir / "manifest_val.csv", index=False)
    test_df.to_csv(output_dir / "manifest_test.csv", index=False)

    manifest["class_name"].value_counts().sort_index().rename_axis("class_name").reset_index(name="count").to_csv(
        output_dir / "class_distribution.csv",
        index=False,
    )


# =========================================================
# 9. INFERENCE CONTRACT
# =========================================================

@dataclass
class TopKPrediction:
    class_name: str
    routing_fruit: str
    confidence: float


@dataclass
class GeneralClassificationOutput:
    module_name: str
    model_version: str
    predicted_class: Optional[str]
    routing_fruit: Optional[str]
    predicted_index: Optional[int]
    confidence: float
    rejected: bool
    skipped: bool
    rejection_reason: Optional[str]
    skip_reason: Optional[str]
    top_k: List[TopKPrediction]
    probability_vector: List[float]
    embedding_vector: Optional[List[float]]
    original_image_size: Tuple[int, int]
    model_input_size: Tuple[int, int]


class GeneralClassifierService:
    def __init__(self, model_path: str, metadata_path: str):
        self.model = tf.keras.models.load_model(model_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.module_name = meta["module_name"]
        self.model_version = meta["model_version"]
        self.image_size = tuple(meta["image_size"])
        self.rejection_threshold = float(meta["rejection_threshold"])
        self.class_to_routing_fruit = meta.get("class_to_routing_fruit", {})

        idx_to_class_raw = meta["idx_to_class"]
        self.idx_to_class = {int(k): v for k, v in idx_to_class_raw.items()}

        self.embedding_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer("embedding").output,
        )

    def _load_single_image(self, image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        original_size = inspect_image_size(image_path)
        image_bytes = tf.io.read_file(image_path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, self.image_size, method=tf.image.ResizeMethod.BILINEAR)
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, axis=0)
        return image.numpy(), original_size

    def predict(
        self,
        image_path: str,
        top_k: int = 3,
        return_embedding: bool = True,
    ) -> GeneralClassificationOutput:
        x, original_size = self._load_single_image(image_path)

        probs = self.model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        pred_class = self.idx_to_class[pred_idx]
        routing_fruit = self.class_to_routing_fruit.get(pred_class, pred_class)

        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_predictions = []
        for i in top_indices:
            class_name = self.idx_to_class[int(i)]
            top_predictions.append(
                TopKPrediction(
                    class_name=class_name,
                    routing_fruit=self.class_to_routing_fruit.get(class_name, class_name),
                    confidence=float(probs[int(i)]),
                )
            )

        rejected = confidence < self.rejection_threshold
        rejection_reason = "low_confidence" if rejected else None

        embedding_vector = None
        if return_embedding:
            embedding_vector = self.embedding_model.predict(x, verbose=0)[0].astype(float).tolist()


	return GeneralClassificationOutput(
            module_name=self.module_name,
            model_version=self.model_version,
            predicted_class=None if rejected else pred_class,
            routing_fruit=None if rejected else routing_fruit,
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
            model_input_size=self.image_size,
        )

    def to_downstream_features(self, output: GeneralClassificationOutput) -> Dict[str, object]:
        return {
            "general_predicted_class": output.predicted_class,
            "general_routing_fruit": output.routing_fruit,
            "general_predicted_index": output.predicted_index,
            "general_confidence": output.confidence,
            "general_rejected": output.rejected,
            "general_probability_vector": output.probability_vector,
        }


# =========================================================
# 10. RUNNER
# =========================================================

def run_training(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)

    manifest = build_manifest(
        data_dir=cfg.data_dir,
        class_to_routing_fruit=cfg.class_to_routing_fruit,
    )
    validate_manifest(manifest, cfg.min_images_per_class)

    class_to_idx, idx_to_class = make_label_mapping(manifest)
    manifest["label"] = manifest["class_name"].map(class_to_idx)

    if cfg.class_to_routing_fruit is None:
        class_to_routing_fruit = {class_name: class_name for class_name in class_to_idx}
    else:
        class_to_routing_fruit = {
            class_name: cfg.class_to_routing_fruit.get(class_name, class_name)
            for class_name in class_to_idx
        }

    train_df, val_df, test_df = stratified_split(
        manifest=manifest,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        seed=cfg.seed,
    )

    train_ds = build_dataset(train_df, cfg.image_size, cfg.batch_size, training=True, seed=cfg.seed)
    val_ds = build_dataset(val_df, cfg.image_size, cfg.batch_size, training=False, seed=cfg.seed)
    test_ds = build_dataset(test_df, cfg.image_size, cfg.batch_size, training=False, seed=cfg.seed)

    model = build_general_classifier(
        num_classes=len(class_to_idx),
        image_size=cfg.image_size,
        dense_units=cfg.dense_units,
        dropout_rate=cfg.dropout_rate,
    )

    model = train_model(model, train_ds, val_ds, cfg, output_dir)

    rejection_threshold = evaluate_and_save_reports(
        model=model,
        val_ds=val_ds,
        test_ds=test_ds,
        idx_to_class=idx_to_class,
        output_dir=output_dir,
        cfg=cfg,
    )

    export_artifacts(
        model=model,
        class_to_idx=class_to_idx,
        class_to_routing_fruit=class_to_routing_fruit,
        rejection_threshold=rejection_threshold,
        cfg=cfg,
        manifest=manifest,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=output_dir,
    )

    print(f"Training complete. Artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    cfg = TrainConfig(
        data_dir=r"C:\Data\DataSetSmallV1\General Classification",
        output_dir="general_classification_runs",
        image_size=(224, 224),
        batch_size=32,
        model_version="general_classification_v2",
        class_to_routing_fruit={
            # examples only - complete this with your exact class names
            "Banana Namwa": "Banana",
            "Banana Plantain": "Banana",
            "Mango Kent": "Mango",
            "Mango Keitt": "Mango",
            "Lime Sweet": "Lime",
        },
    )
    run_training(cfg)
