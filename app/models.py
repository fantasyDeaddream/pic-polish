"""Model registry for PicPolish (single source of truth for model metadata).

This module deliberately has **no torch / PIL dependencies**: it only inspects
weight files on disk, so ``GET /api/models`` can answer in milliseconds without
loading a model or initialising CUDA.

Public API (see ``docs/CONTRACT.md`` §3.7 and ``docs/API.md`` §14)::

    MODEL_REGISTRY: dict[str, dict]        # id -> metadata (insertion order is API order)
    InvalidModelError(ValueError)
    get_model(model_id) -> dict            # copy of the entry + computed "available"
    list_models() -> dict                  # {"default": ..., "models": [...]} for /api/models
    resolve_weights(model_id, ratio, denoise_level=0) -> dict
    available_model_ids() -> list[str]

The engine layer (``app/engine.py``) uses :func:`resolve_weights` to turn a
requested model id + upscale ratio into a concrete checkpoint.  ``"auto"`` is
resolved to Real-ESRGAN x2plus (ratio <= 2) or x4plus (otherwise), exactly like
the pre-multi-model engine did, and degrades to any other available rrdb
checkpoint when the preferred one is missing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import config

__all__ = [
    "MODEL_REGISTRY",
    "DEFAULT_MODEL",
    "InvalidModelError",
    "get_model",
    "list_models",
    "resolve_weights",
    "available_model_ids",
    "weight_paths_of",
]

#: A file smaller than this is treated as a partial download, not a model.
MIN_WEIGHT_BYTES = 1_000_000

DEFAULT_MODEL = "auto"

#: License/source constants (see docs/API.md §14.5)
_LICENSE_ESRGAN = "BSD-3-Clause"
_LICENSE_CUGAN = "MIT"
_SOURCE_ESRGAN = "https://github.com/xinntao/Real-ESRGAN"
_SOURCE_CUGAN = "https://github.com/bilibili/ailab/tree/main/Real-CUGAN"

_RRDB_DENOISE = {"supported": True, "native": False, "levels": [0, 1], "default": 0}
_CUGAN_DENOISE = {"supported": True, "native": True, "levels": [0, 1, 2, 3], "default": 0}
_CUGAN_CONSERVATIVE_DENOISE = {
    "supported": False,
    "native": True,
    "levels": [0],
    "default": 0,
}

_CUGAN_DIR = "realcugan"
_CUGAN_VARIANTS = ("no-denoise", "denoise1x", "denoise2x", "denoise3x", "conservative")
_CUGAN_2X_FILES = {
    "no-denoise": f"{_CUGAN_DIR}/up2x-latest-no-denoise.pth",
    "denoise1x": f"{_CUGAN_DIR}/up2x-latest-denoise1x.pth",
    "denoise2x": f"{_CUGAN_DIR}/up2x-latest-denoise2x.pth",
    "denoise3x": f"{_CUGAN_DIR}/up2x-latest-denoise3x.pth",
    "conservative": f"{_CUGAN_DIR}/up2x-latest-conservative.pth",
}
_CUGAN_3X_FILES = {
    "no-denoise": f"{_CUGAN_DIR}/up3x-latest-no-denoise.pth",
    "denoise3x": f"{_CUGAN_DIR}/up3x-latest-denoise3x.pth",
    "conservative": f"{_CUGAN_DIR}/up3x-latest-conservative.pth",
}
_CUGAN_4X_FILES = {
    "no-denoise": f"{_CUGAN_DIR}/up4x-latest-no-denoise.pth",
    "denoise3x": f"{_CUGAN_DIR}/up4x-latest-denoise3x.pth",
    "conservative": f"{_CUGAN_DIR}/up4x-latest-conservative.pth",
}


# ---------------------------------------------------------------------------
# Registry (insertion order == /api/models order)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: "dict[str, dict]" = {
    "auto": {
        "id": "auto",
        "label": "智能推荐",
        "description": "按目标倍率自动选择 Real-ESRGAN x2plus / x4plus",
        "tags": ["通用"],
        "recommended_for": ["不确定选哪个时"],
        "family": "rrdb",
        "native_scales": [2, 4],
        "num_block": 23,
        # scale -> relative weight path
        "files": {
            2: "RealESRGAN_x2plus.pth",
            4: "RealESRGAN_x4plus.pth",
        },
        "denoise": dict(_RRDB_DENOISE),
        "license": _LICENSE_ESRGAN,
        "size_mb": 127.9,
        "source": _SOURCE_ESRGAN,
    },
    "realesrgan-x4plus": {
        "id": "realesrgan-x4plus",
        "label": "Real-ESRGAN 通用照片",
        "description": "Real-ESRGAN x4plus 通用照片超分（原生 4x，细节自然）",
        "tags": ["通用", "照片"],
        "recommended_for": ["照片", "风景", "人像"],
        "family": "rrdb",
        "native_scales": [4],
        "num_block": 23,
        "files": {4: "RealESRGAN_x4plus.pth"},
        "denoise": dict(_RRDB_DENOISE),
        "license": _LICENSE_ESRGAN,
        "size_mb": 63.9,
        "source": _SOURCE_ESRGAN,
    },
    "realesrgan-x2plus": {
        "id": "realesrgan-x2plus",
        "label": "Real-ESRGAN 低倍率",
        "description": "Real-ESRGAN x2plus 原生 2x，倍率 ≤2 时质量与速度更佳",
        "tags": ["通用", "低倍率"],
        "recommended_for": ["只需要 ≤2 倍放大"],
        "family": "rrdb",
        "native_scales": [2],
        "num_block": 23,
        "files": {2: "RealESRGAN_x2plus.pth"},
        "denoise": dict(_RRDB_DENOISE),
        "license": _LICENSE_ESRGAN,
        "size_mb": 64.0,
        "source": _SOURCE_ESRGAN,
    },
    "realesrgan-anime6b": {
        "id": "realesrgan-anime6b",
        "label": "Real-ESRGAN 动漫",
        "description": "轻量动漫专用模型（RRDB 6 blocks，原生 4x）",
        "tags": ["动漫", "插画"],
        "recommended_for": ["动漫截图", "插画", "二次元"],
        "family": "rrdb",
        "native_scales": [4],
        "num_block": 6,
        "files": {4: "RealESRGAN_x4plus_anime_6B.pth"},
        "denoise": dict(_RRDB_DENOISE),
        "license": _LICENSE_ESRGAN,
        "size_mb": 17.1,
        "source": _SOURCE_ESRGAN,
    },
    "realcugan": {
        "id": "realcugan",
        "label": "Real-CUGAN（动漫/视频）",
        "description": "bilibili Real-CUGAN v3，动漫/视频截图专用，支持原生降噪 0..3",
        "tags": ["动漫", "视频", "降噪"],
        "recommended_for": ["动漫/视频截图", "噪点明显", "需要降噪"],
        "family": "cugan",
        "native_scales": [2, 3, 4],
        "files": {
            2: dict(_CUGAN_2X_FILES),
            3: dict(_CUGAN_3X_FILES),
            4: dict(_CUGAN_4X_FILES),
        },
        "denoise": dict(_CUGAN_DENOISE),
        "license": _LICENSE_CUGAN,
        "size_mb": 55.4,
        "source": _SOURCE_CUGAN,
    },
    "realcugan-conservative": {
        "id": "realcugan-conservative",
        "label": "Real-CUGAN 保守",
        "description": "Real-CUGAN 保守权重，减少过度锐化/风格化（固定策略，不支持降噪等级）",
        "tags": ["动漫", "保守"],
        "recommended_for": ["不想过度锐化/风格化"],
        "family": "cugan",
        "native_scales": [2, 3, 4],
        "files": {
            2: {"conservative": _CUGAN_2X_FILES["conservative"]},
            3: {"conservative": _CUGAN_3X_FILES["conservative"]},
            4: {"conservative": _CUGAN_4X_FILES["conservative"]},
        },
        "denoise": dict(_CUGAN_CONSERVATIVE_DENOISE),
        "license": _LICENSE_CUGAN,
        "size_mb": 15.2,
        "source": _SOURCE_CUGAN,
    },
}


class InvalidModelError(ValueError):
    """Raised for an unknown model id (API maps it to HTTP 400)."""


# ---------------------------------------------------------------------------
# Path / availability helpers
# ---------------------------------------------------------------------------
def _weight_path(relative: str) -> Path:
    """Absolute path of a registry weight entry (resolved at call time)."""
    return Path(config.MODELS_DIR) / relative


def _iter_weight_rel_paths(files) -> list[str]:
    """Flatten the per-family ``files`` structure into relative path strings."""
    if isinstance(files, str):
        return [files]
    if isinstance(files, dict):
        result: list[str] = []
        for value in files.values():
            result.extend(_iter_weight_rel_paths(value))
        return result
    if isinstance(files, (list, tuple)):
        result = []
        for value in files:
            result.extend(_iter_weight_rel_paths(value))
        return result
    return []


def weight_paths_of(entry: dict) -> list[Path]:
    """Every weight file a registry entry may use (absolute paths)."""
    return [_weight_path(rel) for rel in _iter_weight_rel_paths(entry.get("files"))]


def _file_ok(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= MIN_WEIGHT_BYTES
    except OSError:  # pragma: no cover - defensive
        return False


def _entry_available(entry: dict) -> bool:
    """True when at least one weight file of the entry exists and is complete."""
    return any(_file_ok(path) for path in weight_paths_of(entry))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_model(model_id: str) -> dict:
    """Return a copy of a registry entry with a computed ``available`` flag.

    Raises:
        InvalidModelError: unknown ``model_id``.
    """
    if not isinstance(model_id, str) or model_id not in MODEL_REGISTRY:
        options = "/".join(MODEL_REGISTRY)
        raise InvalidModelError(f"未知的 model：{model_id}（可选 {options}）")
    entry = dict(MODEL_REGISTRY[model_id])
    # Nested structures are copied so callers cannot mutate the registry.
    entry["denoise"] = _copy_denoise(MODEL_REGISTRY[model_id]["denoise"])
    entry["tags"] = list(MODEL_REGISTRY[model_id]["tags"])
    entry["recommended_for"] = list(MODEL_REGISTRY[model_id]["recommended_for"])
    entry["native_scales"] = list(MODEL_REGISTRY[model_id]["native_scales"])
    entry["files"] = _copy_files(MODEL_REGISTRY[model_id]["files"])
    entry["available"] = _entry_available(entry)
    return entry


def _copy_files(files):
    if isinstance(files, dict):
        return {key: _copy_files(value) for key, value in files.items()}
    if isinstance(files, list):
        return list(files)
    return files


def _copy_denoise(denoise: dict) -> dict:
    """Deep copy of a ``denoise`` capability descriptor."""
    return {
        "supported": bool(denoise["supported"]),
        "native": bool(denoise["native"]),
        "levels": list(denoise["levels"]),
        "default": int(denoise["default"]),
    }


def _public_model(entry: dict) -> dict:
    """Project a registry entry into the exact /api/models JSON shape."""
    return {
        "id": entry["id"],
        "label": entry["label"],
        "family": entry["family"],
        "native_scales": list(entry["native_scales"]),
        "description": entry["description"],
        "tags": list(entry["tags"]),
        "recommended_for": list(entry["recommended_for"]),
        "denoise": _copy_denoise(entry["denoise"]),
        "license": entry["license"],
        "size_mb": entry["size_mb"],
        "available": _entry_available(entry),
        "source": entry["source"],
    }


def list_models() -> dict:
    """Return the ``GET /api/models`` payload (fixed order, no model loading)."""
    return {
        "default": DEFAULT_MODEL,
        "models": [_public_model(entry) for entry in MODEL_REGISTRY.values()],
    }


def available_model_ids() -> list[str]:
    """Ids whose weight files are present, in registry order."""
    return [model_id for model_id, entry in MODEL_REGISTRY.items() if _entry_available(entry)]


# ---------------------------------------------------------------------------
# Weight resolution (model id + upscale ratio + denoise level -> checkpoint)
# ---------------------------------------------------------------------------
def _require_file(relative: str, model_id: str) -> Path:
    path = _weight_path(relative)
    if not _file_ok(path):
        raise FileNotFoundError(
            f"模型 {model_id} 的权重缺失：{relative}，请运行 scripts/fetch_models.py 下载"
        )
    return path


def _normalise_denoise_level(denoise_level: int) -> int:
    try:
        level = int(denoise_level)
    except (TypeError, ValueError):
        level = 0
    return max(0, min(3, level))


def _pick_cugan_scale(ratio: float) -> int:
    """Native CUGAN scale for an upscale ratio: <=2 -> 2, <=3 -> 3, else 4."""
    if ratio <= 2.0 + 1e-9:
        return 2
    if ratio <= 3.0 + 1e-9:
        return 3
    return 4


def _cugan_variant_for(scale: int, denoise_level: int) -> str:
    """Map a native denoise level to the CUGAN weight variant name."""
    if denoise_level <= 0:
        return "no-denoise"
    if scale == 2:
        return {1: "denoise1x", 2: "denoise2x", 3: "denoise3x"}.get(denoise_level, "denoise3x")
    # 3x / 4x only ship a denoise3x variant; levels 1..3 all use it.
    return "denoise3x"


def _resolve_rrdb(entry: dict, model_id: str, ratio: float) -> dict:
    files = entry["files"]  # scale -> relpath
    scales = sorted(int(scale) for scale in files)
    if model_id == "auto":
        preferred = 2 if ratio <= 2.0 + 1e-9 else 4
        order = [preferred] + [scale for scale in (4, 2) if scale != preferred]
        if preferred not in scales:
            order = [scale for scale in (2, 4) if scale in scales] or scales
    else:
        preferred = entry["native_scales"][0]
        order = [preferred] + [scale for scale in scales if scale != preferred]
    for scale in order:
        if scale not in files:
            continue
        path = _weight_path(files[scale])
        if _file_ok(path):
            return {
                "model_id": model_id,
                "label": entry["label"],
                "family": "rrdb",
                "native_scale": int(scale),
                "weight_paths": [path],
                "weight_file": files[scale],
                "num_block": int(entry.get("num_block", 23)),
                "denoise_level": 0,  # rrdb denoise is OpenCV pre-processing, not a weight
            }
    missing = ", ".join(files[scale] for scale in scales if scale in files)
    raise FileNotFoundError(
        f"模型 {model_id} 的权重缺失：{missing}，请运行 scripts/fetch_models.py 下载"
    )


def _cugan_variant_level(variant: str, requested_level: int) -> int:
    """Effective denoise level for a concrete CUGAN weight variant."""
    return {"no-denoise": 0, "conservative": 0, "denoise1x": 1, "denoise2x": 2,
            "denoise3x": 3}.get(variant, requested_level)


def _resolve_cugan(entry: dict, model_id: str, ratio: float, denoise_level: int) -> dict:
    conservative = model_id == "realcugan-conservative"
    requested_level = 0 if conservative else _normalise_denoise_level(denoise_level)
    files = entry["files"]
    scales = sorted(int(key) for key in files)
    if not scales:  # pragma: no cover - registry always ships 2/3/4
        raise FileNotFoundError(
            f"模型 {model_id} 的权重缺失，请运行 scripts/fetch_models.py 下载"
        )
    preferred = _pick_cugan_scale(float(ratio))
    # Try the ideal native scale first, then the nearest existing other scales.
    order = [preferred] + sorted(
        (scale for scale in scales if scale != preferred),
        key=lambda scale: (abs(scale - preferred), -scale),
    )
    for scale in order:
        variants = files.get(scale) or {}
        if not variants:
            continue
        wanted = "conservative" if conservative else _cugan_variant_for(scale, requested_level)
        candidate_names = [wanted, "no-denoise"] + [
            name for name in variants if name not in (wanted, "no-denoise")
        ]
        for name in candidate_names:
            relative = variants.get(name)
            if relative is None or not _file_ok(_weight_path(relative)):
                continue
            level = requested_level if name == wanted else _cugan_variant_level(name, requested_level)
            return {
                "model_id": model_id,
                "label": entry["label"],
                "family": "cugan",
                "native_scale": int(scale),
                "weight_paths": [_weight_path(relative)],
                "weight_file": relative,
                "variant": name,
                "denoise_level": int(level),
            }
    expected = ", ".join(
        relative for variants in files.values() for relative in variants.values()
    )
    raise FileNotFoundError(
        f"模型 {model_id} 的权重缺失：{expected}，请运行 scripts/fetch_models.py 下载"
    )


def resolve_weights(model_id: str, ratio: float, denoise_level: int = 0) -> dict:
    """Resolve a requested model id to a concrete checkpoint + native scale.

    Args:
        model_id: one of :data:`MODEL_REGISTRY` (``"auto"`` included).
        ratio: requested upscale ratio (``max(out_w/src_w, out_h/src_h)``);
            only used to pick the native scale.
        denoise_level: Real-CUGAN native denoise level ``0..3`` (ignored by rrdb
            models and by ``realcugan-conservative``).

    Returns:
        ``{"model_id", "label", "family", "native_scale", "weight_paths",
        "weight_file", "denoise_level"}`` plus family extras (``num_block`` for
        rrdb, ``variant`` for CUGAN).

    Raises:
        InvalidModelError: unknown model id.
        FileNotFoundError: a required weight file is missing/incomplete.
    """
    entry = get_model(model_id)
    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        ratio = 1.0
    if ratio <= 0:
        ratio = 1.0
    if entry["family"] == "cugan":
        return _resolve_cugan(entry, model_id, ratio, denoise_level)
    return _resolve_rrdb(entry, model_id, ratio)
