from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from general_classification_module import GeneralClassifierService
from variety_classification_module import VarietyClassifierService
from quality_classification_module import QualityClassifierService
from ripeness_stage_module import RipenessStageService
from storage_condition_module import StorageConditionService
from shelf_life_model import ShelfLifeService


# =========================================================
# 1. CONFIG
# =========================================================

@dataclass
class PipelinePaths:
    general_model_path: str
    general_metadata_path: str
    variety_registry_path: str
    quality_registry_path: str
    ripeness_registry_path: str
    storage_registry_path: str
    shelf_life_model_dir: str


@dataclass
class InferenceContext:
    day: Optional[int] = None
    storage_temp_c: Optional[float] = None
    storage_humidity_rh: Optional[float] = None


@dataclass
class EndToEndPipelineOutput:
    pipeline_version: str
    image_path: str
    original_image_size: Optional[Tuple[int, int]]
    routing_fruit: Optional[str]
    variety_name: Optional[str]
    quality_label: Optional[str]
    ripeness_stage: Optional[str]
    storage_condition: Optional[str]
    shelf_life_days: Dict[str, Optional[float]]
    module_status: Dict[str, Dict[str, Any]]
    latencies_ms: Dict[str, float]
    module_outputs: Dict[str, Dict[str, Any]]
    pipeline_status: str
    message: str


# =========================================================
# 2. LOW-LEVEL HELPERS
# =========================================================

def ensure_exists(path_str: str, label: str) -> None:
    if not Path(path_str).exists():
        raise FileNotFoundError(f"{label} not found: {path_str}")


def inspect_image_size(image_path: str) -> Tuple[int, int]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    return int(image.shape[0]), int(image.shape[1])


def to_plain_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    raise TypeError(f"Unsupported output type for normalization: {type(obj)}")


def maybe_get_original_size(*outputs: Dict[str, Any], fallback_image_path: Optional[str] = None) -> Optional[Tuple[int, int]]:
    for output in outputs:
        value = output.get("original_image_size")
        if value is not None:
            return tuple(value)
    if fallback_image_path is not None:
        try:
            return inspect_image_size(fallback_image_path)
        except Exception:
            return None
    return None


def run_timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, round(elapsed_ms, 2)


# =========================================================
# 3. NORMALIZATION HELPERS
# =========================================================

def normalize_general_output(raw_output: Any) -> Dict[str, Any]:
    raw = to_plain_dict(raw_output)
    routing_fruit = raw.get("routing_fruit") or raw.get("predicted_class")
    return {
        "module_name": raw.get("module_name", "general_classification"),
        "model_version": raw.get("model_version"),
        "routing_fruit": routing_fruit,
        "predicted_index": raw.get("predicted_index"),
        "confidence": float(raw.get("confidence", 0.0) or 0.0),
        "rejected": bool(raw.get("rejected", False)),
        "skipped": bool(raw.get("skipped", False)),
        "rejection_reason": raw.get("rejection_reason"),
        "skip_reason": raw.get("skip_reason"),
        "top_k": raw.get("top_k", []),
        "probability_vector": raw.get("probability_vector", []),
        "embedding_vector": raw.get("embedding_vector"),
        "original_image_size": raw.get("original_image_size"),
        "model_input_size": raw.get("model_input_size"),
    }


def normalize_variety_output(raw_output: Any) -> Dict[str, Any]:
    raw = to_plain_dict(raw_output)
    return {
        "module_name": raw.get("module_name", "variety_classification"),
        "model_version": raw.get("model_version"),
        "routing_fruit": raw.get("routing_fruit") or raw.get("general_fruit"),
        "predicted_variety": raw.get("predicted_variety"),
        "predicted_index": raw.get("predicted_index"),
        "confidence": float(raw.get("confidence", 0.0) or 0.0),
        "rejected": bool(raw.get("rejected", False)),
        "skipped": bool(raw.get("skipped", False)),
        "rejection_reason": raw.get("rejection_reason"),
        "skip_reason": raw.get("skip_reason"),
        "top_k": raw.get("top_k", []),
        "probability_vector": raw.get("probability_vector", []),
        "embedding_vector": raw.get("embedding_vector"),
        "original_image_size": raw.get("original_image_size"),
        "model_input_size": raw.get("model_input_size"),
    }


def normalize_quality_output(raw_output: Any) -> Dict[str, Any]:
    raw = to_plain_dict(raw_output)
    return {
        "module_name": raw.get("module_name", "quality_classification"),
        "model_version": raw.get("model_version"),
        "routing_fruit": raw.get("routing_fruit") or raw.get("general_fruit"),
        "variety_name": raw.get("variety_name"),
        "predicted_quality": raw.get("predicted_quality"),
        "quality_score_good": float(raw.get("quality_score_good", 0.0) or 0.0),
        "quality_score_bad": float(raw.get("quality_score_bad", 0.0) or 0.0),
        "confidence": float(raw.get("confidence", 0.0) or 0.0),
        "rejected": bool(raw.get("rejected", False)),
        "skipped": bool(raw.get("skipped", False)),
        "rejection_reason": raw.get("rejection_reason"),
        "skip_reason": raw.get("skip_reason"),
        "decision_threshold": raw.get("decision_threshold"),
        "rejection_threshold": raw.get("rejection_threshold"),
        "embedding_vector": raw.get("embedding_vector"),
        "original_image_size": raw.get("original_image_size"),
        "model_input_size": raw.get("model_input_size"),
    }


def normalize_ripeness_output(raw_output: Any) -> Dict[str, Any]:
    raw = to_plain_dict(raw_output)
    return {
        "module_name": raw.get("module_name", "ripeness_stage_classification"),
        "model_version": raw.get("model_version"),
        "routing_fruit": raw.get("routing_fruit") or raw.get("general_fruit"),
        "variety_name": raw.get("variety_name"),
        "quality_label": raw.get("quality_label"),
        "predicted_stage": raw.get("predicted_stage"),
        "predicted_index": raw.get("predicted_index"),
        "confidence": float(raw.get("confidence", 0.0) or 0.0),
        "rejected": bool(raw.get("rejected", False)),
        "skipped": bool(raw.get("skipped", False)),
        "rejection_reason": raw.get("rejection_reason"),
        "skip_reason": raw.get("skip_reason"),
        "top_k": raw.get("top_k", []),
        "probability_vector": raw.get("probability_vector", []),
        "embedding_vector": raw.get("embedding_vector"),
        "original_image_size": raw.get("original_image_size"),
        "model_input_size": raw.get("model_input_size"),
    }


def normalize_storage_output(raw_output: Any) -> Dict[str, Any]:
    raw = to_plain_dict(raw_output)
    return {
        "module_name": raw.get("module_name", "storage_condition_classification"),
        "model_version": raw.get("model_version"),
        "routing_fruit": raw.get("routing_fruit") or raw.get("general_fruit"),
        "variety_name": raw.get("variety_name"),
        "quality_label": raw.get("quality_label"),
        "ripeness_stage": raw.get("ripeness_stage"),
        "predicted_storage_condition": raw.get("predicted_storage_condition"),
        "predicted_index": raw.get("predicted_index"),
        "confidence": float(raw.get("confidence", 0.0) or 0.0),
        "rejected": bool(raw.get("rejected", False)),
        "skipped": bool(raw.get("skipped", False)),
        "rejection_reason": raw.get("rejection_reason"),
        "skip_reason": raw.get("skip_reason"),
        "top_k": raw.get("top_k", []),
        "probability_vector": raw.get("probability_vector", []),
        "embedding_vector": raw.get("embedding_vector"),
        "original_image_size": raw.get("original_image_size"),
        "model_input_size": raw.get("model_input_size"),
    }


def normalize_shelf_life_output(raw_output: Any) -> Dict[str, Any]:
    raw = to_plain_dict(raw_output)
    return {
        "module_name": raw.get("module_name", "shelf_life_prediction"),
        "model_version": raw.get("model_version"),
        "routing_fruit": raw.get("routing_fruit"),
        "variety_name": raw.get("variety_name"),
        "quality_label": raw.get("quality_label"),
        "ripeness_stage": raw.get("ripeness_stage"),
        "storage_condition": raw.get("storage_condition"),
        "day": raw.get("day"),
        "storage_temp_c": raw.get("storage_temp_c"),
        "storage_humidity_rh": raw.get("storage_humidity_rh"),
        "predicted_days_to_m2": raw.get("predicted_days_to_m2"),
        "predicted_days_to_m3": raw.get("predicted_days_to_m3"),
        "predicted_days_to_m4": raw.get("predicted_days_to_m4"),
        "predicted_days_to_r": raw.get("predicted_days_to_r"),
        "raw_prediction_vector": raw.get("raw_prediction_vector", []),
        "rejected": bool(raw.get("rejected", False)),
        "skipped": bool(raw.get("skipped", False)),
        "rejection_reason": raw.get("rejection_reason"),
        "skip_reason": raw.get("skip_reason"),
        "embedding_vector": raw.get("embedding_vector"),
        "original_image_size": raw.get("original_image_size"),
        "model_input_size": raw.get("model_input_size"),
    }


# =========================================================
# 4. BRIDGE DICTS
# =========================================================

def build_general_bridge(out: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "module_name": out["module_name"],
        "model_version": out["model_version"],
        "routing_fruit": out["routing_fruit"],
        "predicted_index": out["predicted_index"],
        "confidence": out["confidence"],
        "rejected": out["rejected"],
        "skipped": out["skipped"],
        "rejection_reason": out["rejection_reason"],
        "skip_reason": out["skip_reason"],
        "probability_vector": out["probability_vector"],
    }


def build_variety_bridge(out: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "module_name": out["module_name"],
        "model_version": out["model_version"],
        "routing_fruit": out["routing_fruit"],
        "predicted_variety": out["predicted_variety"],
        "predicted_index": out["predicted_index"],
        "confidence": out["confidence"],
        "rejected": out["rejected"],
        "skipped": out["skipped"],
        "rejection_reason": out["rejection_reason"],
        "skip_reason": out["skip_reason"],
        "probability_vector": out["probability_vector"],
    }


def build_quality_bridge(out: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "module_name": out["module_name"],
        "model_version": out["model_version"],
        "routing_fruit": out["routing_fruit"],
        "variety_name": out["variety_name"],
        "predicted_quality": out["predicted_quality"],
        "confidence": out["confidence"],
        "rejected": out["rejected"],
        "skipped": out["skipped"],
    }


def build_ripeness_bridge(out: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "module_name": out["module_name"],
        "model_version": out["model_version"],
        "routing_fruit": out["routing_fruit"],
        "variety_name": out["variety_name"],
        "quality_label": out["quality_label"],
        "predicted_stage": out["predicted_stage"],
        "predicted_index": out["predicted_index"],
        "confidence": out["confidence"],
        "rejected": out["rejected"],
        "skipped": out["skipped"],
        "probability_vector": out["probability_vector"],
    }


def build_storage_bridge(out: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "module_name": out["module_name"],
        "model_version": out["model_version"],
        "routing_fruit": out["routing_fruit"],
        "variety_name": out["variety_name"],
        "quality_label": out["quality_label"],
        "ripeness_stage": out["ripeness_stage"],
        "predicted_storage_condition": out["predicted_storage_condition"],
        "predicted_index": out["predicted_index"],
        "confidence": out["confidence"],
        "rejected": out["rejected"],
        "skipped": out["skipped"],
        "probability_vector": out["probability_vector"],
    }


# =========================================================
# 5. STATUS HELPERS
# =========================================================

def module_status_block(output: Dict[str, Any], accepted_label: Optional[str]) -> Dict[str, Any]:
    return {
        "accepted_label": accepted_label,
        "confidence": output.get("confidence"),
        "rejected": output.get("rejected", False),
        "skipped": output.get("skipped", False),
        "rejection_reason": output.get("rejection_reason"),
        "skip_reason": output.get("skip_reason"),
    }


def build_pipeline_status(out_map: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    if out_map["general"].get("rejected"):
        return "rejected", "Pipeline stopped: general classification rejected."
    if out_map["shelf_life"] and not out_map["shelf_life"].get("skipped", True):
        return "complete", "Pipeline completed successfully."
    return "partial", "Pipeline finished partial execution."


# =========================================================
# 6. ORCHESTRATOR
# =========================================================

class FruitInspectionPipeline:
    def __init__(self, paths: PipelinePaths):
        self.general_service = GeneralClassifierService(paths.general_model_path, paths.general_metadata_path)
        self.variety_service = VarietyClassifierService(paths.variety_registry_path)
        self.quality_service = QualityClassifierService(paths.quality_registry_path)
        self.ripeness_service = RipenessStageService(paths.ripeness_registry_path)
        self.storage_service = StorageConditionService(paths.storage_registry_path)
        self.shelf_life_service = ShelfLifeService(paths.shelf_life_model_dir)

    def predict(self, image_path: str, context: Optional[InferenceContext] = None, top_k: int = 3, return_embeddings: bool = False) -> EndToEndPipelineOutput:
        ctx = context or InferenceContext()
        latencies: Dict[str, float] = {}

        # 1. Pipeline execution
        general_raw, general_latency = run_timed(
            self.general_service.predict,
            image_path,
            top_k=top_k,
            return_embedding=return_embeddings,
        )
        general_out = normalize_general_output(general_raw)
        latencies["general"] = general_latency
        gen_bridge = build_general_bridge(general_out)

        variety_raw, variety_latency = run_timed(
            self.variety_service.predict,
            image_path,
            gen_bridge,
            top_k=top_k,
            return_embedding=return_embeddings,
        )
        variety_out = normalize_variety_output(variety_raw)
        latencies["variety"] = variety_latency
        var_bridge = build_variety_bridge(variety_out)

        quality_raw, quality_latency = run_timed(
            self.quality_service.predict,
            image_path,
            gen_bridge,
            var_bridge,
            return_embedding=return_embeddings,
        )
        quality_out = normalize_quality_output(quality_raw)
        latencies["quality"] = quality_latency
        qual_bridge = build_quality_bridge(quality_out)

        ripeness_raw, ripeness_latency = run_timed(
            self.ripeness_service.predict,
            image_path,
            gen_bridge,
            var_bridge,
            qual_bridge,
            top_k=top_k,
            return_embedding=return_embeddings,
        )
        ripeness_out = normalize_ripeness_output(ripeness_raw)
        latencies["ripeness"] = ripeness_latency
        ripe_bridge = build_ripeness_bridge(ripeness_out)

        storage_raw, storage_latency = run_timed(
            self.storage_service.predict,
            image_path,
            gen_bridge,
            var_bridge,
            qual_bridge,
            ripe_bridge,
            top_k=top_k,
            return_embedding=return_embeddings,
        )
        storage_out = normalize_storage_output(storage_raw)
        latencies["storage"] = storage_latency
        stor_bridge = build_storage_bridge(storage_out)

        shelf_raw, shelf_latency = run_timed(
            self.shelf_life_service.predict,
            image_path,
            gen_bridge,
            var_bridge,
            qual_bridge,
            ripe_bridge,
            stor_bridge,
            ctx.day,
            ctx.storage_temp_c,
            ctx.storage_humidity_rh,
            return_embedding=return_embeddings,
        )
        shelf_out = normalize_shelf_life_output(shelf_raw)
        latencies["shelf_life"] = shelf_latency

        out_map = {"general": general_out, "variety": variety_out, "quality": quality_out, "ripeness": ripeness_out, "storage": storage_out, "shelf_life": shelf_out}
        status, message = build_pipeline_status(out_map)

        return EndToEndPipelineOutput(
            pipeline_version="pipeline_v1",
            image_path=image_path,
            original_image_size=maybe_get_original_size(*out_map.values(), fallback_image_path=image_path),
            routing_fruit=general_out.get("routing_fruit"),
            variety_name=variety_out.get("predicted_variety"),
            quality_label=quality_out.get("predicted_quality"),
            ripeness_stage=ripeness_out.get("predicted_stage"),
            storage_condition=storage_out.get("predicted_storage_condition"),
            shelf_life_days={"m2": shelf_out.get("predicted_days_to_m2"), "m3": shelf_out.get("predicted_days_to_m3"), "m4": shelf_out.get("predicted_days_to_m4"), "r": shelf_out.get("predicted_days_to_r")},
            module_status={
                "general": module_status_block(general_out, general_out.get("routing_fruit")),
                "variety": module_status_block(variety_out, variety_out.get("predicted_variety")),
                "quality": module_status_block(quality_out, quality_out.get("predicted_quality")),
                "ripeness": module_status_block(ripeness_out, ripeness_out.get("predicted_stage")),
                "storage": module_status_block(storage_out, storage_out.get("predicted_storage_condition")),
                "shelf_life": module_status_block(
                    shelf_out,
                    "predicted"
                    if not shelf_out.get("rejected", False) and not shelf_out.get("skipped", False)
                    else None,
                ),
            },
            latencies_ms=latencies,
            module_outputs=out_map,
            pipeline_status=status,
            message=message
        )

if __name__ == "__main__":
    pass
