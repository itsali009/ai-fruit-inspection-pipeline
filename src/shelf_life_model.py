from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit


AUTOTUNE = tf.data.AUTOTUNE
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

STAGE_ORDER = ["M1", "M2", "M3", "M4", "R"]
STAGE_TO_INDEX = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}
TARGET_STAGES = ["M2", "M3", "M4", "R"]
TARGET_COLUMNS = [f"days_to_{stage}" for stage in TARGET_STAGES]

FILENAME_PATTERN = re.compile(
    r"^(?P<fruit>[A-Za-z]+)(?P<storage_condition>[HSW])_day(?P<day>\d+)_IMG(?P<img_number>\d+)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)


# =========================================================
# 1. CONFIG
# =========================================================

@dataclass
class ShelfLifeTrainConfig:
    excel_path: str
    image_root: str
    output_dir: str = "shelf_life_runs"

    image_size: int = 224
    batch_size: int = 32
    seed: int = 42

    epochs_stage1: int = 20
    epochs_stage2: int = 8
    learning_rate_stage1: float = 1e-3
    learning_rate_stage2: float = 1e-5

    validation_size: float = 0.15
    test_size: float = 0.15

    pretrained: bool = True
    fine_tune: bool = True
    unfreeze_last_n_layers: int = 30

    dense_units_image: int = 256
    dense_units_metadata: int = 64
    dense_units_fusion_1: int = 256
    dense_units_fusion_2: int = 128
    dropout_image: float = 0.25
    dropout_metadata: float = 0.10
    dropout_fusion: float = 0.25

    model_version: str = "shelf_life_v2"


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
    return int(image.shape[0]), int(image.shape[1])


# =========================================================
# 3. FILE PARSING / METADATA PREP
# =========================================================

def parse_filename(filename: str) -> Dict[str, object]:
    clean_name = Path(str(filename).strip()).name
    match = FILENAME_PATTERN.match(clean_name)

    if not match:
        raise ValueError(
            f"Filename does not match expected pattern: {clean_name}\n"
            "Expected example: MangoH_day14_IMG36.jpg"
        )

    routing_fruit = match.group("fruit").capitalize()
    storage_condition = match.group("storage_condition").upper()
    day = int(match.group("day"))
    img_number = int(match.group("img_number"))
    trajectory_id = f"{routing_fruit}{storage_condition}_IMG{img_number}"

    return {
        "filename": clean_name,
        "parsed_routing_fruit": routing_fruit,
        "storage_condition_from_filename": storage_condition,
        "day_from_filename": day,
        "img_number": img_number,
        "trajectory_id": trajectory_id,
    }


def load_and_clean_metadata(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    df.columns = [str(col).strip() for col in df.columns]

    required_columns = [
        "Image Names",
        "fruit_type",
        "Day",
        "storage_temp(°C)",
        "storage_humidity(RH%)",
        "ripeness_stage",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Excel file: {missing}")

    df = df[required_columns].copy()
    df["Image Names"] = df["Image Names"].astype(str).str.strip().apply(lambda x: Path(x).name)
    df["fruit_type"] = df["fruit_type"].astype(str).str.strip().str.capitalize()
    df["ripeness_stage"] = df["ripeness_stage"].astype(str).str.strip().str.upper()

    parsed = df["Image Names"].apply(parse_filename).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    df["day"] = df["day_from_filename"]

    mismatched_fruit = df[df["fruit_type"] != df["parsed_routing_fruit"]]
    if not mismatched_fruit.empty:
        raise ValueError(
            f"Found {len(mismatched_fruit)} rows where Excel fruit_type and filename fruit do not match."
        )

    invalid_stages = sorted(set(df["ripeness_stage"]) - set(STAGE_TO_INDEX))
    if invalid_stages:
        raise ValueError(f"Unknown ripeness stages found: {invalid_stages}")

    df["stage_index"] = df["ripeness_stage"].map(STAGE_TO_INDEX)

    df = df.rename(
        columns={
            "Image Names": "filename",
            "fruit_type": "routing_fruit",
            "storage_temp(°C)": "storage_temp_c",
            "storage_humidity(RH%)": "storage_humidity_rh",
        }
    )

    df["storage_condition"] = df["storage_condition_from_filename"]

    return df


def build_image_index(image_root: str) -> Dict[str, str]:
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")

    image_index: Dict[str, str] = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        key = path.name
        if key in image_index:
            raise ValueError(
                f"Duplicate filename detected across folders: {key}\n"
                "Base filenames must be unique for filename matching."
            )
        image_index[key] = str(path.resolve())

    if not image_index:
        raise ValueError(f"No images found under: {root}")

    return image_index


def attach_image_paths(df: pd.DataFrame, image_index: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["image_path"] = df["filename"].map(image_index)

    missing_images = df[df["image_path"].isna()]
    if not missing_images.empty:
        sample = missing_images["filename"].head(10).tolist()
        raise ValueError(
            f"{len(missing_images)} filenames from Excel could not be found in the image folder.\n"
            f"Examples: {sample}"
        )

    return df


# =========================================================
# 4. TARGET ENGINEERING
# =========================================================

def build_threshold_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for target_col in TARGET_COLUMNS:
        df[target_col] = np.nan

    enriched_groups = []

    for _, group in df.groupby("trajectory_id", sort=False):
        group = group.sort_values("day").copy()

        days = group["day"].to_numpy()
        stage_idx = group["stage_index"].to_numpy()

        for row_pos in range(len(group)):
            current_day = days[row_pos]

            for target_stage in TARGET_STAGES:
                target_idx = STAGE_TO_INDEX[target_stage]
                eligible_days = days[(days >= current_day) & (stage_idx >= target_idx)]
                target_value = np.nan if len(eligible_days) == 0 else int(eligible_days.min() - current_day)
                group.iloc[row_pos, group.columns.get_loc(f"days_to_{target_stage}")] = target_value

        enriched_groups.append(group)

    result = pd.concat(enriched_groups, ignore_index=True)
    result = result.dropna(subset=TARGET_COLUMNS, how="all").reset_index(drop=True)
    return result


# =========================================================
# 5. GROUPED SPLIT
# =========================================================

def grouped_train_val_test_split(
    df: pd.DataFrame,
    validation_size: float,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df["trajectory_id"].to_numpy()

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(df, groups=groups))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    relative_val_size = validation_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["trajectory_id"].to_numpy()))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


# =========================================================
# 6. METADATA ENCODING
# =========================================================

def fit_metadata_encoder(train_df: pd.DataFrame) -> Dict[str, object]:
    routing_fruit_vocab = sorted(train_df["routing_fruit"].unique().tolist())
    storage_condition_vocab = sorted(train_df["storage_condition"].unique().tolist())
    ripeness_vocab = STAGE_ORDER[:]

    numeric_columns = ["day", "storage_temp_c", "storage_humidity_rh"]
    numeric_stats = {}

    for col in numeric_columns:
        mean = float(train_df[col].mean())
        std = float(train_df[col].std(ddof=0))
        numeric_stats[col] = {"mean": mean, "std": std if std > 0 else 1.0}

    return {
        "routing_fruit_vocab": routing_fruit_vocab,
        "storage_condition_vocab": storage_condition_vocab,
        "ripeness_vocab": ripeness_vocab,
        "numeric_stats": numeric_stats,
    }


def _one_hot(values: List[str], vocab: List[str], feature_name: str) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(vocab)}
    output = np.zeros((len(values), len(vocab)), dtype=np.float32)

    for row_idx, value in enumerate(values):
        if value not in index:
            raise ValueError(f"Value '{value}' not seen during encoder fitting for feature '{feature_name}'.")
        output[row_idx, index[value]] = 1.0

    return output


def transform_metadata(df: pd.DataFrame, encoder: Dict[str, object]) -> np.ndarray:
    routing_fruit_ohe = _one_hot(
        df["routing_fruit"].tolist(),
        encoder["routing_fruit_vocab"],
        "routing_fruit",
    )
    storage_condition_ohe = _one_hot(
        df["storage_condition"].tolist(),
        encoder["storage_condition_vocab"],
        "storage_condition",
    )
    ripeness_ohe = _one_hot(
        df["ripeness_stage"].tolist(),
        encoder["ripeness_vocab"],
        "ripeness_stage",
    )

    numeric_parts = []
    for col, stats in encoder["numeric_stats"].items():
        values = df[col].to_numpy(dtype=np.float32)
        normalized = (values - stats["mean"]) / stats["std"]
        numeric_parts.append(normalized.reshape(-1, 1))

    numeric_matrix = np.concatenate(numeric_parts, axis=1).astype(np.float32)
    metadata_matrix = np.concatenate(
        [routing_fruit_ohe, storage_condition_ohe, ripeness_ohe, numeric_matrix],
        axis=1,
    )
    return metadata_matrix.astype(np.float32)


# =========================================================
# 7. TF.DATA
# =========================================================

def decode_and_resize_image(path: tf.Tensor, image_size: int) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32)
    return image


def make_dataset(
    df: pd.DataFrame,
    encoder: Dict[str, object],
    image_size: int,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    image_paths = df["image_path"].astype(str).to_numpy()
    metadata = transform_metadata(df, encoder)
    labels = df[TARGET_COLUMNS].to_numpy(dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices(
        (
            {
                "image_path": image_paths,
                "metadata": metadata,
            },
            labels,
        )
    )

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    def _map_fn(inputs, labels_):
        image = decode_and_resize_image(inputs["image_path"], image_size)
        return {"image": image, "metadata": inputs["metadata"]}, labels_

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# =========================================================
# 8. MASKED LOSS / METRICS
# =========================================================

def masked_huber(y_true: tf.Tensor, y_pred: tf.Tensor, delta: float = 1.0) -> tf.Tensor:
    mask = tf.cast(tf.math.is_finite(y_true), tf.float32)
    safe_y_true = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))

    error = safe_y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic

    loss = 0.5 * tf.square(quadratic) + delta * linear
    loss = loss * mask

    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)


def masked_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    mask = tf.cast(tf.math.is_finite(y_true), tf.float32)
    safe_y_true = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))
    error = tf.abs(safe_y_true - y_pred) * mask
    return tf.reduce_sum(error) / (tf.reduce_sum(mask) + 1e-8)


def get_custom_objects() -> Dict[str, object]:
    return {
        "masked_huber": masked_huber,
        "masked_mae": masked_mae,
    }


# =========================================================
# 9. MODEL
# =========================================================

def build_model(config: ShelfLifeTrainConfig, metadata_dim: int) -> tf.keras.Model:
    image_input = tf.keras.Input(shape=(config.image_size, config.image_size, 3), name="image")
    metadata_input = tf.keras.Input(shape=(metadata_dim,), name="metadata")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.05),
        ],
        name="augmentation",
    )

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet" if config.pretrained else None,
        input_shape=(config.image_size, config.image_size, 3),
        name="backbone",
    )
    backbone.trainable = False

    x = augmentation(image_input)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dense(config.dense_units_image, activation="relu", name="image_dense")(x)
    x = tf.keras.layers.Dropout(config.dropout_image, name="image_dropout")(x)

    m = tf.keras.layers.Dense(config.dense_units_metadata, activation="relu", name="metadata_dense")(metadata_input)
    m = tf.keras.layers.Dropout(config.dropout_metadata, name="metadata_dropout")(m)

    fused = tf.keras.layers.Concatenate(name="fusion_concat")([x, m])
    fused = tf.keras.layers.Dense(config.dense_units_fusion_1, activation="relu", name="fusion_dense_1")(fused)
    fused = tf.keras.layers.Dropout(config.dropout_fusion, name="fusion_dropout")(fused)
    embedding = tf.keras.layers.Dense(config.dense_units_fusion_2, activation="relu", name="embedding")(fused)

    outputs = tf.keras.layers.Dense(len(TARGET_COLUMNS), name="shelf_life_outputs")(embedding)

    return tf.keras.Model(
        inputs=[image_input, metadata_input],
        outputs=outputs,
        name="shelf_life_model",
    )


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=masked_huber,
        metrics=[masked_mae],
    )


def unfreeze_backbone_tail(model: tf.keras.Model, unfreeze_last_n_layers: int) -> None:
    backbone = model.get_layer("backbone")
    backbone.trainable = True

    if unfreeze_last_n_layers > 0:
        for layer in backbone.layers[:-unfreeze_last_n_layers]:
            layer.trainable = False

    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


# =========================================================
# 10. EVALUATION
# =========================================================

def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def evaluate_predictions(df: pd.DataFrame, predictions: np.ndarray) -> Dict[str, object]:
    y_true = df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    results: Dict[str, object] = {
        "overall": {},
        "per_routing_fruit": {},
        "per_storage_condition": {},
    }

    target_maes = {}
    for idx, target in enumerate(TARGET_COLUMNS):
        target_maes[target] = _safe_mae(y_true[:, idx], predictions[:, idx])

    results["overall"]["target_mae"] = target_maes
    results["overall"]["average_mae"] = float(np.nanmean(list(target_maes.values())))

    for routing_fruit, fruit_df in df.groupby("routing_fruit"):
        fruit_preds = predictions[fruit_df.index]
        fruit_true = fruit_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)

        fruit_target_maes = {
            target: _safe_mae(fruit_true[:, i], fruit_preds[:, i])
            for i, target in enumerate(TARGET_COLUMNS)
        }
        results["per_routing_fruit"][routing_fruit] = {
            "target_mae": fruit_target_maes,
            "average_mae": float(np.nanmean(list(fruit_target_maes.values()))),
        }

    for storage_condition, env_df in df.groupby("storage_condition"):
        env_preds = predictions[env_df.index]
        env_true = env_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)

        env_target_maes = {
            target: _safe_mae(env_true[:, i], env_preds[:, i])
            for i, target in enumerate(TARGET_COLUMNS)
        }
        results["per_storage_condition"][storage_condition] = {
            "target_mae": env_target_maes,
            "average_mae": float(np.nanmean(list(env_target_maes.values()))),
        }

    return results


# =========================================================
# 11. TRAINING PIPELINE
# =========================================================

def train(config: ShelfLifeTrainConfig) -> None:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    ensure_dir(output_dir)

    print("Loading metadata...")
    df = load_and_clean_metadata(config.excel_path)

    print("Indexing image files...")
    image_index = build_image_index(config.image_root)
    df = attach_image_paths(df, image_index)

    print("Building threshold-based shelf-life targets...")
    df = build_threshold_targets(df)

    print("Performing grouped train/val/test split...")
    train_df, val_df, test_df = grouped_train_val_test_split(
        df=df,
        validation_size=config.validation_size,
        test_size=config.test_size,
        seed=config.seed,
    )

    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Unique trajectories in train: {train_df['trajectory_id'].nunique()}")
    print(f"Unique trajectories in val: {val_df['trajectory_id'].nunique()}")
    print(f"Unique trajectories in test: {test_df['trajectory_id'].nunique()}")

    encoder = fit_metadata_encoder(train_df)
    save_json(encoder, output_dir / "metadata_encoder.json")
    save_json(asdict(config), output_dir / "config.json")

    train_ds = make_dataset(train_df, encoder, config.image_size, config.batch_size, training=True, seed=config.seed)
    val_ds = make_dataset(val_df, encoder, config.image_size, config.batch_size, training=False, seed=config.seed)
    test_ds = make_dataset(test_df, encoder, config.image_size, config.batch_size, training=False, seed=config.seed)

    metadata_dim = transform_metadata(train_df.head(1), encoder).shape[1]

    print("Building model...")
    model = build_model(config, metadata_dim)
    compile_model(model, config.learning_rate_stage1)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_shelf_life_model.keras"),
            monitor="val_masked_mae",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_masked_mae",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_masked_mae",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ]

    print("Starting warm-up training...")
    warmup_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs_stage1,
        callbacks=callbacks,
        verbose=1,
    )

    history_bundle = {"warmup": warmup_history.history}

    if config.fine_tune and config.epochs_stage2 > 0:
        print("Starting fine-tuning...")
        unfreeze_backbone_tail(model, config.unfreeze_last_n_layers)
        compile_model(model, config.learning_rate_stage2)

        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs_stage1 + config.epochs_stage2,
            initial_epoch=config.epochs_stage1,
            callbacks=callbacks,
            verbose=1,
        )
        history_bundle["fine_tune"] = fine_tune_history.history

    save_json(history_bundle, output_dir / "history.json")

    print("Reloading best model for evaluation...")
    best_model = tf.keras.models.load_model(
        output_dir / "best_shelf_life_model.keras",
        custom_objects=get_custom_objects(),
    )

    print("Running test predictions...")
    predictions = best_model.predict(test_ds, verbose=1)
    predictions = np.maximum(predictions, 0.0)
    metrics = evaluate_predictions(test_df.reset_index(drop=True), predictions)

    save_json(metrics, output_dir / "metrics.json")

    model_metadata = {
        "module_name": "shelf_life_prediction",
        "model_version": config.model_version,
        "image_size": config.image_size,
        "target_columns": TARGET_COLUMNS,
        "target_stages": TARGET_STAGES,
        "routing_fruit_vocab": encoder["routing_fruit_vocab"],
        "storage_condition_vocab": encoder["storage_condition_vocab"],
        "ripeness_vocab": encoder["ripeness_vocab"],
    }
    save_json(model_metadata, output_dir / "model_metadata.json")

    train_df.to_csv(output_dir / "manifest_train.csv", index=False)
    val_df.to_csv(output_dir / "manifest_val.csv", index=False)
    test_df.to_csv(output_dir / "manifest_test.csv", index=False)

    print("\n=== Final Test Metrics ===")
    print(json.dumps(metrics["overall"], indent=2))
    print(f"\nArtifacts saved to: {output_dir.resolve()}")


# =========================================================
# 12. INFERENCE CONTRACT
# =========================================================

@dataclass
class ShelfLifeOutput:
    module_name: str
    model_version: str
    routing_fruit: Optional[str]
    variety_name: Optional[str]
    quality_label: Optional[str]
    ripeness_stage: Optional[str]
    storage_condition: Optional[str]
    day: Optional[int]
    storage_temp_c: Optional[float]
    storage_humidity_rh: Optional[float]
    predicted_days_to_m2: Optional[float]
    predicted_days_to_m3: Optional[float]
    predicted_days_to_m4: Optional[float]
    predicted_days_to_r: Optional[float]
    raw_prediction_vector: List[float]
    rejected: bool
    skipped: bool
    rejection_reason: Optional[str]
    skip_reason: Optional[str]
    embedding_vector: Optional[List[float]]
    original_image_size: Optional[Tuple[int, int]]
    model_input_size: Optional[Tuple[int, int]]


class ShelfLifeService:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)

        with open(self.model_dir / "metadata_encoder.json", "r", encoding="utf-8") as f:
            self.encoder = json.load(f)

        with open(self.model_dir / "model_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.model = tf.keras.models.load_model(
            self.model_dir / "best_shelf_life_model.keras",
            custom_objects=get_custom_objects(),
        )

        self.embedding_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer("embedding").output,
        )

    def _load_single_image(self, image_path: str, image_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        original_size = inspect_image_size(image_path)
        image = decode_and_resize_image(tf.constant(image_path), image_size)
        image = tf.expand_dims(image, axis=0)
        return image.numpy(), original_size

    def _prepare_single_metadata_vector(
        self,
        routing_fruit: str,
        storage_condition: str,
        day: int,
        storage_temp_c: float,
        storage_humidity_rh: float,
        ripeness_stage: str,
    ) -> np.ndarray:
        row = pd.DataFrame(
            [
                {
                    "routing_fruit": routing_fruit.capitalize(),
                    "storage_condition": str(storage_condition).upper(),
                    "ripeness_stage": str(ripeness_stage).upper(),
                    "day": int(day),
                    "storage_temp_c": float(storage_temp_c),
                    "storage_humidity_rh": float(storage_humidity_rh),
                }
            ]
        )
        return transform_metadata(row, self.encoder).astype(np.float32)

    def predict(
        self,
        image_path: str,
        general_output: Dict,
        variety_output: Optional[Dict] = None,
        quality_output: Optional[Dict] = None,
        ripeness_output: Optional[Dict] = None,
        storage_output: Optional[Dict] = None,
        day: Optional[int] = None,
        storage_temp_c: Optional[float] = None,
        storage_humidity_rh: Optional[float] = None,
        return_embedding: bool = True,
    ) -> ShelfLifeOutput:
        general_rejected = bool(general_output.get("rejected", False))
        general_skipped = bool(general_output.get("skipped", False))
        routing_fruit = general_output.get("routing_fruit")

        variety_name = None if variety_output is None else variety_output.get("predicted_variety")
        quality_label = None if quality_output is None else quality_output.get("predicted_quality")

        ripeness_rejected = bool(False if ripeness_output is None else ripeness_output.get("rejected", False))
        ripeness_skipped = bool(False if ripeness_output is None else ripeness_output.get("skipped", False))
        ripeness_stage = None if ripeness_output is None else ripeness_output.get("predicted_stage")

        storage_rejected = bool(False if storage_output is None else storage_output.get("rejected", False))
        storage_skipped = bool(False if storage_output is None else storage_output.get("skipped", False))
        predicted_storage_condition = None if storage_output is None else storage_output.get("predicted_storage_condition")

        if general_rejected or general_skipped or routing_fruit is None:
            return ShelfLifeOutput(
                module_name="shelf_life_prediction",
                model_version=self.metadata["model_version"],
                routing_fruit=routing_fruit,
                variety_name=variety_name,
                quality_label=quality_label,
                ripeness_stage=ripeness_stage,
                storage_condition=predicted_storage_condition,
                day=day,
                storage_temp_c=storage_temp_c,
                storage_humidity_rh=storage_humidity_rh,
                predicted_days_to_m2=None,
                predicted_days_to_m3=None,
                predicted_days_to_m4=None,
                predicted_days_to_r=None,
                raw_prediction_vector=[],
                rejected=False,
                skipped=True,
                rejection_reason=None,
                skip_reason="general_classification_rejected_skipped_or_missing_routing",
                embedding_vector=None,
                original_image_size=None,
                model_input_size=None,
            )

        if ripeness_output is None or ripeness_rejected or ripeness_skipped or ripeness_stage is None:
            return ShelfLifeOutput(
                module_name="shelf_life_prediction",
                model_version=self.metadata["model_version"],
                routing_fruit=routing_fruit,
                variety_name=variety_name,
                quality_label=quality_label,
                ripeness_stage=ripeness_stage,
                storage_condition=predicted_storage_condition,
                day=day,
                storage_temp_c=storage_temp_c,
                storage_humidity_rh=storage_humidity_rh,
                predicted_days_to_m2=None,
                predicted_days_to_m3=None,
                predicted_days_to_m4=None,
                predicted_days_to_r=None,
                raw_prediction_vector=[],
                rejected=False,
                skipped=True,
                rejection_reason=None,
                skip_reason="ripeness_output_missing_rejected_or_skipped",
                embedding_vector=None,
                original_image_size=None,
                model_input_size=None,
            )

        if storage_output is None or storage_rejected or storage_skipped or predicted_storage_condition is None:
            return ShelfLifeOutput(
                module_name="shelf_life_prediction",
                model_version=self.metadata["model_version"],
                routing_fruit=routing_fruit,
                variety_name=variety_name,
                quality_label=quality_label,
                ripeness_stage=ripeness_stage,
                storage_condition=predicted_storage_condition,
                day=day,
                storage_temp_c=storage_temp_c,
                storage_humidity_rh=storage_humidity_rh,
                predicted_days_to_m2=None,
                predicted_days_to_m3=None,
                predicted_days_to_m4=None,
                predicted_days_to_r=None,
                raw_prediction_vector=[],
                rejected=False,
                skipped=True,
                rejection_reason=None,
                skip_reason="storage_output_missing_rejected_or_skipped",
                embedding_vector=None,
                original_image_size=None,
                model_input_size=None,
            )

        if day is None or storage_temp_c is None or storage_humidity_rh is None:
            return ShelfLifeOutput(
                module_name="shelf_life_prediction",
                model_version=self.metadata["model_version"],
                routing_fruit=routing_fruit,
                variety_name=variety_name,
                quality_label=quality_label,
                ripeness_stage=ripeness_stage,
                storage_condition=predicted_storage_condition,
                day=day,
                storage_temp_c=storage_temp_c,
                storage_humidity_rh=storage_humidity_rh,
                predicted_days_to_m2=None,
                predicted_days_to_m3=None,
                predicted_days_to_m4=None,
                predicted_days_to_r=None,
                raw_prediction_vector=[],
                rejected=False,
                skipped=True,
                rejection_reason=None,
                skip_reason="missing_required_numeric_metadata",
                embedding_vector=None,
                original_image_size=None,
                model_input_size=None,
            )

        try:
            metadata_vector = self._prepare_single_metadata_vector(
                routing_fruit=routing_fruit,
                storage_condition=predicted_storage_condition,
                day=int(day),
                storage_temp_c=float(storage_temp_c),
                storage_humidity_rh=float(storage_humidity_rh),
                ripeness_stage=ripeness_stage,
            )
        except ValueError as e:
            return ShelfLifeOutput(
                module_name="shelf_life_prediction",
                model_version=self.metadata["model_version"],
                routing_fruit=routing_fruit,
                variety_name=variety_name,
                quality_label=quality_label,
                ripeness_stage=ripeness_stage,
                storage_condition=predicted_storage_condition,
                day=day,
                storage_temp_c=storage_temp_c,
                storage_humidity_rh=storage_humidity_rh,
                predicted_days_to_m2=None,
                predicted_days_to_m3=None,
                predicted_days_to_m4=None,
                predicted_days_to_r=None,
                raw_prediction_vector=[],
                rejected=False,
                skipped=True,
                rejection_reason=None,
                skip_reason=f"metadata_encoding_failed: {str(e)}",
                embedding_vector=None,
                original_image_size=None,
                model_input_size=None,
            )

        image_size = int(self.metadata["image_size"])
        x_image, original_size = self._load_single_image(image_path, image_size)

        prediction = self.model.predict(
            {"image": x_image, "metadata": metadata_vector},
            verbose=0,
        )[0]
        prediction = np.maximum(prediction.astype(float), 0.0)

        embedding_vector = None
        if return_embedding:
            embedding_vector = (
                self.embedding_model.predict({"image": x_image, "metadata": metadata_vector}, verbose=0)[0]
                .astype(float)
                .tolist()
            )

        return ShelfLifeOutput(
            module_name="shelf_life_prediction",
            model_version=self.metadata["model_version"],
            routing_fruit=routing_fruit,
            variety_name=variety_name,
            quality_label=quality_label,
            ripeness_stage=ripeness_stage,
            storage_condition=predicted_storage_condition,
            day=int(day),
            storage_temp_c=float(storage_temp_c),
            storage_humidity_rh=float(storage_humidity_rh),
            predicted_days_to_m2=float(prediction[0]),
            predicted_days_to_m3=float(prediction[1]),
            predicted_days_to_m4=float(prediction[2]),
            predicted_days_to_r=float(prediction[3]),
            raw_prediction_vector=prediction.tolist(),
            rejected=False,
            skipped=False,
            rejection_reason=None,
            skip_reason=None,
            embedding_vector=embedding_vector,
            original_image_size=original_size,
            model_input_size=(image_size, image_size),
        )

    def to_pipeline_features(self, output: ShelfLifeOutput) -> Dict[str, object]:
        return {
            "shelf_life_routing_fruit": output.routing_fruit,
            "shelf_life_variety_name": output.variety_name,
            "shelf_life_quality_label": output.quality_label,
            "shelf_life_ripeness_stage": output.ripeness_stage,
            "shelf_life_storage_condition": output.storage_condition,
            "shelf_life_day": output.day,
            "shelf_life_storage_temp_c": output.storage_temp_c,
            "shelf_life_storage_humidity_rh": output.storage_humidity_rh,
            "shelf_life_days_to_m2": output.predicted_days_to_m2,
            "shelf_life_days_to_m3": output.predicted_days_to_m3,
            "shelf_life_days_to_m4": output.predicted_days_to_m4,
            "shelf_life_days_to_r": output.predicted_days_to_r,
            "shelf_life_rejected": output.rejected,
            "shelf_life_skipped": output.skipped,
            "shelf_life_raw_prediction_vector": output.raw_prediction_vector,
        }


# =========================================================
# 13. CLI
# =========================================================

def parse_args() -> ShelfLifeTrainConfig:
    parser = argparse.ArgumentParser(description="Train shelf-life prediction model.")
    parser.add_argument("--excel-path", type=str, required=True, help="Path to Shelf Life V1.xlsx")
    parser.add_argument("--image-root", type=str, required=True, help="Root folder containing fruit images")
    parser.add_argument("--output-dir", type=str, default="shelf_life_runs")

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-stage1", type=int, default=20)
    parser.add_argument("--epochs-stage2", type=int, default=8)

    parser.add_argument("--learning-rate-stage1", type=float, default=1e-3)
    parser.add_argument("--learning-rate-stage2", type=float, default=1e-5)

    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--unfreeze-last-n-layers", type=int, default=30)

    args = parser.parse_args()

    return ShelfLifeTrainConfig(
        excel_path=args.excel_path,
        image_root=args.image_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        learning_rate_stage1=args.learning_rate_stage1,
        learning_rate_stage2=args.learning_rate_stage2,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
        pretrained=args.pretrained,
        fine_tune=args.fine_tune,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
    )


if __name__ == "__main__":
    # Simplified main block for module saving
    pass
