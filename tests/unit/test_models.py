"""Unit tests for :mod:`app.models` (multi-model registry).

The registry is pure metadata + filesystem probing: tests use a temporary
"models" directory with truncated dummy files (size-only validity check), so
they run identically on machines without the real weights.  Only the final
"real weights" smoke test is skipped when ``models/`` is empty.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # allow `pytest tests/unit/...` from any cwd
    sys.path.insert(0, str(ROOT))

from app import config, models

MODEL_IDS = [
    "auto",
    "realesrgan-x4plus",
    "realesrgan-x2plus",
    "realesrgan-anime6b",
    "realcugan",
    "realcugan-conservative",
]

#: Exact /api/models field set (docs/API.md §14.1).
PUBLIC_FIELDS = {
    "id",
    "label",
    "family",
    "native_scales",
    "description",
    "tags",
    "recommended_for",
    "denoise",
    "license",
    "size_mb",
    "available",
    "source",
}

#: Internal registry field set (docs/CONTRACT.md §3.7 + task-7).
REGISTRY_FIELDS = {
    "id",
    "label",
    "description",
    "tags",
    "recommended_for",
    "family",
    "native_scales",
    "files",
    "denoise",
    "license",
    "size_mb",
    "source",
}


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _fake_file(path: Path, size: int = models.MIN_WEIGHT_BYTES + 4096) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.truncate(size)  # sparse: instant, size-only validity check


@pytest.fixture()
def fake_models_dir(monkeypatch, tmp_path):
    """A models/ directory containing every registered weight (dummy content)."""
    root = tmp_path / "models"
    monkeypatch.setattr(config, "MODELS_DIR", root)
    for entry in models.MODEL_REGISTRY.values():
        for path in models.weight_paths_of(entry):
            _fake_file(path)
    return root


@pytest.fixture()
def empty_models_dir(monkeypatch, tmp_path):
    root = tmp_path / "models_empty"
    root.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", root)
    return root


# ---------------------------------------------------------------------------
# Registry structure
# ---------------------------------------------------------------------------
def test_registry_order_and_default():
    assert list(models.MODEL_REGISTRY) == MODEL_IDS
    assert models.DEFAULT_MODEL == "auto"


def test_registry_required_fields():
    for model_id, entry in models.MODEL_REGISTRY.items():
        assert set(entry) >= REGISTRY_FIELDS, model_id
        assert entry["id"] == model_id
        assert isinstance(entry["label"], str) and entry["label"]
        assert isinstance(entry["description"], str) and entry["description"]
        assert isinstance(entry["tags"], list) and entry["tags"]
        assert isinstance(entry["recommended_for"], list) and entry["recommended_for"]
        assert entry["family"] in ("rrdb", "cugan")
        assert all(isinstance(scale, int) and scale in (2, 3, 4)
                   for scale in entry["native_scales"])
        assert entry["license"] in ("BSD-3-Clause", "MIT")
        assert isinstance(entry["size_mb"], float) and entry["size_mb"] > 0
        assert entry["source"].startswith("https://github.com/")
        assert entry["files"], model_id


def test_registry_families():
    expected = {
        "auto": "rrdb",
        "realesrgan-x4plus": "rrdb",
        "realesrgan-x2plus": "rrdb",
        "realesrgan-anime6b": "rrdb",
        "realcugan": "cugan",
        "realcugan-conservative": "cugan",
    }
    assert {mid: entry["family"] for mid, entry in models.MODEL_REGISTRY.items()} == expected


def test_registry_native_scales():
    assert models.MODEL_REGISTRY["auto"]["native_scales"] == [2, 4]
    assert models.MODEL_REGISTRY["realesrgan-x4plus"]["native_scales"] == [4]
    assert models.MODEL_REGISTRY["realesrgan-x2plus"]["native_scales"] == [2]
    assert models.MODEL_REGISTRY["realesrgan-anime6b"]["native_scales"] == [4]
    assert models.MODEL_REGISTRY["realcugan"]["native_scales"] == [2, 3, 4]
    assert models.MODEL_REGISTRY["realcugan-conservative"]["native_scales"] == [2, 3, 4]


def test_registry_denoise_semantics():
    for model_id in ("auto", "realesrgan-x4plus", "realesrgan-x2plus", "realesrgan-anime6b"):
        denoise = models.MODEL_REGISTRY[model_id]["denoise"]
        assert denoise["supported"] is True
        assert denoise["native"] is False
        assert denoise["levels"] == [0, 1]
        assert denoise["default"] == 0
    cugan = models.MODEL_REGISTRY["realcugan"]["denoise"]
    assert cugan == {"supported": True, "native": True, "levels": [0, 1, 2, 3], "default": 0}
    conservative = models.MODEL_REGISTRY["realcugan-conservative"]["denoise"]
    assert conservative["levels"] == [0]
    assert conservative["supported"] is False


def test_anime6b_uses_six_blocks():
    assert models.MODEL_REGISTRY["realesrgan-anime6b"]["num_block"] == 6
    for model_id in ("auto", "realesrgan-x4plus", "realesrgan-x2plus"):
        assert models.MODEL_REGISTRY[model_id]["num_block"] == 23


# ---------------------------------------------------------------------------
# list_models() / get_model()
# ---------------------------------------------------------------------------
def test_list_models_api_shape():
    payload = models.list_models()
    assert payload["default"] == "auto"
    assert [item["id"] for item in payload["models"]] == MODEL_IDS
    for item in payload["models"]:
        assert set(item) == PUBLIC_FIELDS
        assert isinstance(item["available"], bool)
        assert isinstance(item["native_scales"], list)
        assert set(item["denoise"]) == {"supported", "native", "levels", "default"}
        assert isinstance(item["size_mb"], (int, float))


def test_list_models_does_not_import_torch():
    """``/api/models`` must stay fast: no torch import / no model loading."""
    code = (
        "import sys; from app import models; payload = models.list_models();"
        " print('torch' in sys.modules); print(len(payload['models']))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "False"
    assert lines[1] == str(len(MODEL_IDS))
    # Static belt-and-braces check: the module never imports torch.
    source = Path(models.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source


def test_get_model_returns_isolated_copy():
    entry = models.get_model("realcugan")
    assert entry["id"] == "realcugan"
    assert isinstance(entry["available"], bool)
    entry["label"] = "mutated"
    entry["denoise"]["levels"].append(99)
    entry["tags"].append("mutated")
    assert models.MODEL_REGISTRY["realcugan"]["label"] != "mutated"
    assert models.MODEL_REGISTRY["realcugan"]["denoise"]["levels"] == [0, 1, 2, 3]
    assert "mutated" not in models.MODEL_REGISTRY["realcugan"]["tags"]


@pytest.mark.parametrize("bad", [None, "", "x4plus", "RealESRGAN-x4plus", "foo", 42])
def test_get_model_invalid(bad):
    with pytest.raises(models.InvalidModelError) as info:
        models.get_model(bad)
    assert issubclass(models.InvalidModelError, ValueError)
    assert "未知的 model" in str(info.value)
    assert "realcugan" in str(info.value)  # lists the options


# ---------------------------------------------------------------------------
# Availability (file existence only)
# ---------------------------------------------------------------------------
def test_available_model_ids_all_present(fake_models_dir):
    assert models.available_model_ids() == MODEL_IDS
    payload = models.list_models()
    assert all(item["available"] for item in payload["models"])


def test_available_model_ids_empty(empty_models_dir):
    assert models.available_model_ids() == []
    payload = models.list_models()
    assert not any(item["available"] for item in payload["models"])


def test_available_ignores_tiny_partial_files(monkeypatch, tmp_path):
    root = tmp_path / "models_tiny"
    monkeypatch.setattr(config, "MODELS_DIR", root)
    path = root / "RealESRGAN_x4plus.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"partial")
    assert "realesrgan-x4plus" not in models.available_model_ids()
    assert models.get_model("realesrgan-x4plus")["available"] is False


def test_weight_paths_of():
    auto_paths = models.weight_paths_of(models.get_model("auto"))
    assert len(auto_paths) == 2
    assert {path.name for path in auto_paths} == {
        "RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth"
    }
    assert all(path.parent == Path(config.MODELS_DIR) for path in auto_paths)
    cugan_paths = models.weight_paths_of(models.get_model("realcugan"))
    assert len(cugan_paths) == 11
    assert all(path.parent.name == "realcugan" for path in cugan_paths)


# ---------------------------------------------------------------------------
# resolve_weights(): rrdb family
# ---------------------------------------------------------------------------
def test_resolve_auto_scale_by_ratio(fake_models_dir):
    cases = [
        (1.0, 2, "RealESRGAN_x2plus.pth"),
        (1.5, 2, "RealESRGAN_x2plus.pth"),
        (2.0, 2, "RealESRGAN_x2plus.pth"),
        (2.0001, 4, "RealESRGAN_x4plus.pth"),
        (3.0, 4, "RealESRGAN_x4plus.pth"),
        (4.0, 4, "RealESRGAN_x4plus.pth"),
    ]
    for ratio, scale, filename in cases:
        resolved = models.resolve_weights("auto", ratio)
        assert resolved["model_id"] == "auto"
        assert resolved["family"] == "rrdb"
        assert resolved["native_scale"] == scale
        assert Path(resolved["weight_file"]).name == filename
        assert resolved["weight_paths"][0].name == filename
        assert resolved["weight_paths"][0].is_file()
        assert resolved["num_block"] == 23
        assert resolved["denoise_level"] == 0


def test_resolve_explicit_rrdb_models(fake_models_dir):
    x4 = models.resolve_weights("realesrgan-x4plus", 1.5)
    assert (x4["native_scale"], Path(x4["weight_file"]).name) == (4, "RealESRGAN_x4plus.pth")
    assert x4["num_block"] == 23

    x2 = models.resolve_weights("realesrgan-x2plus", 4.0)
    assert (x2["native_scale"], Path(x2["weight_file"]).name) == (2, "RealESRGAN_x2plus.pth")

    anime = models.resolve_weights("realesrgan-anime6b", 1.5)
    assert anime["native_scale"] == 4
    assert Path(anime["weight_file"]).name == "RealESRGAN_x4plus_anime_6B.pth"
    assert anime["num_block"] == 6


def test_resolve_rrdb_ignores_denoise_level(fake_models_dir):
    resolved = models.resolve_weights("realesrgan-x4plus", 1.5, denoise_level=3)
    assert resolved["denoise_level"] == 0  # rrdb denoise is OpenCV preprocessing


# ---------------------------------------------------------------------------
# resolve_weights(): Real-CUGAN family
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ratio,level,scale,variant,effective_level",
    [
        (1.0, 0, 2, "no-denoise", 0),
        (1.5, 0, 2, "no-denoise", 0),
        (1.5, 1, 2, "denoise1x", 1),
        (1.5, 2, 2, "denoise2x", 2),
        (1.5, 3, 2, "denoise3x", 3),
        (2.0, 0, 2, "no-denoise", 0),
        (2.5, 0, 3, "no-denoise", 0),
        (2.5, 2, 3, "denoise3x", 2),
        (3.0, 3, 3, "denoise3x", 3),
        (3.0, 1, 3, "denoise3x", 1),
        (3.1, 0, 4, "no-denoise", 0),
        (4.0, 3, 4, "denoise3x", 3),
    ],
)
def test_resolve_realcugan_matrix(fake_models_dir, ratio, level, scale, variant, effective_level):
    resolved = models.resolve_weights("realcugan", ratio, level)
    assert resolved["family"] == "cugan"
    assert resolved["native_scale"] == scale
    assert resolved["variant"] == variant
    assert resolved["denoise_level"] == effective_level
    assert resolved["weight_file"] == f"realcugan/up{scale}x-latest-{variant}.pth"
    assert resolved["weight_paths"][0].is_file()


@pytest.mark.parametrize(
    "ratio,level,scale",
    [(1.5, 0, 2), (1.5, 3, 2), (3.0, 0, 3), (3.0, 3, 3), (4.0, 1, 4)],
)
def test_resolve_conservative_is_fixed(fake_models_dir, ratio, level, scale):
    resolved = models.resolve_weights("realcugan-conservative", ratio, level)
    assert resolved["variant"] == "conservative"
    assert resolved["native_scale"] == scale
    assert resolved["denoise_level"] == 0  # conservative ignores levels
    assert resolved["weight_file"] == f"realcugan/up{scale}x-latest-conservative.pth"


def test_resolve_weights_invalid_model(fake_models_dir):
    with pytest.raises(models.InvalidModelError):
        models.resolve_weights("not-a-model", 1.5)


# ---------------------------------------------------------------------------
# Missing / partial weights
# ---------------------------------------------------------------------------
def test_missing_weights_raise_with_hint(empty_models_dir):
    with pytest.raises(FileNotFoundError) as info:
        models.resolve_weights("auto", 1.5)
    assert "权重缺失" in str(info.value)
    assert "scripts/fetch_models.py" in str(info.value)
    with pytest.raises(FileNotFoundError):
        models.resolve_weights("realcugan", 1.5)


def test_auto_falls_back_to_available_rrdb(monkeypatch, tmp_path):
    root = tmp_path / "models_partial"
    monkeypatch.setattr(config, "MODELS_DIR", root)
    _fake_file(root / "RealESRGAN_x4plus.pth")
    assert models.resolve_weights("auto", 1.5)["weight_file"] == "RealESRGAN_x4plus.pth"
    assert models.resolve_weights("auto", 3.0)["weight_file"] == "RealESRGAN_x4plus.pth"
    with pytest.raises(FileNotFoundError):
        models.resolve_weights("realesrgan-x2plus", 1.5)


def test_partial_cugan_falls_back_to_available_scale(monkeypatch, tmp_path):
    root = tmp_path / "models_cugan_partial"
    monkeypatch.setattr(config, "MODELS_DIR", root)
    _fake_file(root / "realcugan" / "up2x-latest-no-denoise.pth")
    # ratio 3.0 would prefer 3x, but only the 2x no-denoise weight exists.
    resolved = models.resolve_weights("realcugan", 3.0, 0)
    assert resolved["native_scale"] == 2
    assert resolved["variant"] == "no-denoise"
    # A missing denoise variant degrades to no-denoise and reports level 0.
    resolved = models.resolve_weights("realcugan", 3.0, 2)
    assert resolved["variant"] == "no-denoise"
    assert resolved["denoise_level"] == 0
    assert models.available_model_ids() == ["realcugan"]


# ---------------------------------------------------------------------------
# Real weights smoke test (skipped on machines without models/)
# ---------------------------------------------------------------------------
def test_real_registry_weights_are_complete():
    expected = []
    for model_id, entry in models.MODEL_REGISTRY.items():
        paths = models.weight_paths_of(entry)
        if paths and all(path.is_file() and path.stat().st_size >= models.MIN_WEIGHT_BYTES
                         for path in paths):
            expected.append(model_id)
    if not Path(config.MODELS_DIR).is_dir():
        pytest.skip("models/ 目录不存在")
    assert models.available_model_ids() == expected
    if expected:
        # Every available model must resolve to existing files.
        for model_id in expected:
            resolved = models.resolve_weights(model_id, 2.0, 0)
            assert all(path.is_file() for path in resolved["weight_paths"])
