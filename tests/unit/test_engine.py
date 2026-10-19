"""Unit tests for the PicPolish inference engine (CONTRACT §3.4).

Layers:

* **always run** – PIL backend behaviour, engine factory/failover, model
  resolution helpers, tiling/OOM-retry helpers (no torch, no weights, no GPU);
* **needs torch + weights** – strict ``RRDBNet`` / Real-CUGAN checkpoint loading,
  CPU forward shapes, engine model-resolution matrix (fake model files);
* **needs a GPU** – real inference for the whole model matrix (auto, x4plus,
  x2plus, anime6b, realcugan 0..3, conservative), OpenCV denoise effect,
  Real-CUGAN uint8 handling + native denoise differences, the multi-tile
  black-block regression tests, and the 720p→1080p / 4K deadlines.

Tests never touch the network.  GPU/CPU sections are skipped automatically when
torch, CUDA or the corresponding weights are missing.
"""
from __future__ import annotations

import sys
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # allow `pytest tests/unit/...` from any cwd
    sys.path.insert(0, str(ROOT))

from app import config, engine as engine_mod, models

LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
MODELS_DIR = Path(config.MODELS_DIR)
MODEL_X4 = MODELS_DIR / config.MODEL_X4
MODEL_X2 = MODELS_DIR / config.MODEL_X2
MODEL_ANIME = MODELS_DIR / "RealESRGAN_x4plus_anime_6B.pth"
CUGAN_DIR = MODELS_DIR / "realcugan"
MODEL_IDS = list(models.MODEL_REGISTRY)


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
def _torch():
    """The torch module, or None (same lazy helper the engine uses)."""
    return engine_mod._import_torch()


def _gpu_available() -> bool:
    torch = _torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - broken CUDA install
        return False


def _model_ready(model_id: str) -> bool:
    try:
        return bool(models.get_model(model_id).get("available"))
    except Exception:  # pragma: no cover - defensive
        return False


TORCH_READY = _torch() is not None
GPU_READY = TORCH_READY and _gpu_available()
ANY_MODEL_READY = any(_model_ready(model_id) for model_id in MODEL_IDS)


def requires_model(model_id: str):
    return pytest.mark.skipif(
        not (TORCH_READY and _model_ready(model_id)),
        reason=f"需要 torch + {model_id} 权重",
    )


def requires_gpu_model(model_id: str):
    return pytest.mark.skipif(
        not (GPU_READY and _model_ready(model_id)),
        reason=f"需要 GPU + torch + {model_id} 权重",
    )


requires_gpu = pytest.mark.skipif(not GPU_READY, reason="需要 torch + CUDA")
requires_weights = pytest.mark.skipif(
    not (TORCH_READY and ANY_MODEL_READY), reason="需要 torch + 至少一个模型权重"
)


@pytest.fixture(autouse=True)
def _isolated_engine_cache():
    """Every test starts and ends with an empty engine cache (no GPU leak)."""
    engine_mod.shutdown_engines()
    yield
    engine_mod.shutdown_engines()


@pytest.fixture()
def fake_models_dir(monkeypatch, tmp_path):
    """A models/ dir containing every registry file (dummy sizes, no loading)."""
    root = tmp_path / "models"
    monkeypatch.setattr(config, "MODELS_DIR", root)
    for entry in models.MODEL_REGISTRY.values():
        for path in models.weight_paths_of(entry):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as handle:
                handle.truncate(models.MIN_WEIGHT_BYTES + 4096)
    return root


# ---------------------------------------------------------------------------
# Image / metric helpers
# ---------------------------------------------------------------------------
def make_rgb(width: int = 96, height: int = 64, color=(40, 80, 120)) -> Image.Image:
    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)
    for x in range(0, width, max(4, width // 8)):
        draw.line([(x, 0), (x, height)], fill=(210, 170, 90), width=1)
    draw.ellipse(
        [width // 5, height // 6, width * 4 // 5, height * 5 // 6], fill=(230, 120, 60)
    )
    draw.rectangle([width // 2, height // 3, width * 3 // 4, height * 2 // 3],
                   fill=(60, 150, 200))
    return img


def make_photo_like(width: int = 1280, height: int = 720) -> Image.Image:
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        draw.line(
            [(0, y), (width, y)],
            fill=(30 + y * 180 // height, 80 + y * 80 // height,
                  160 - y * 120 // height),
        )
    for x in range(0, width, 37):
        draw.line([(x, 0), (x, height)], fill=(200, 210, 190), width=1)
    draw.ellipse([width // 5, height // 6, width // 2, height // 2], fill=(240, 120, 60))
    draw.text((width // 10, height * 3 // 4), "PicPolish detail TEXT", fill=(255, 255, 240))
    return img.filter(ImageFilter.GaussianBlur(0.8))


def make_rgba(width: int = 96, height: int = 64) -> Image.Image:
    img = make_rgb(width, height).convert("RGBA")
    alpha = Image.linear_gradient("L").resize((width, height))
    img.putalpha(alpha)
    return img


def make_gradient(width: int = 700, height: int = 700) -> Image.Image:
    """Very gentle gradient: flat enough that block means must stay uniform."""
    x = np.linspace(124.0, 132.0, width, dtype=np.float32)
    y = np.linspace(124.0, 132.0, height, dtype=np.float32)
    arr = np.empty((height, width, 3), dtype=np.uint8)
    arr[..., 0] = np.clip(np.floor(x)[None, :], 0, 255).astype(np.uint8)
    arr[..., 1] = np.clip(np.floor(y)[:, None], 0, 255).astype(np.uint8)
    arr[..., 2] = 128
    return Image.fromarray(arr, "RGB")


def make_noisy(width: int = 320, height: int = 240, sigma: float = 20.0) -> Image.Image:
    rng = np.random.default_rng(0)
    base = np.asarray(make_photo_like(width, height).resize((width, height), LANCZOS))
    noisy = np.clip(base.astype(np.int16) + rng.normal(0, sigma, base.shape).astype(np.int16),
                    0, 255).astype(np.uint8)
    return Image.fromarray(noisy, "RGB")


def sample_photo():
    path = ROOT / "samples" / "photo_1280x720.jpg"
    if path.is_file():
        image = Image.open(path).convert("RGB")
        if image.size != (1280, 720):
            image = image.resize((1280, 720), LANCZOS)
        return image, True
    return make_photo_like(1280, 720), False


def black_fraction(img: Image.Image, threshold: int = 10) -> float:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return float((arr.max(axis=2) < threshold).mean())


def max_block_mean_deviation(img: Image.Image, block: int = 32) -> float:
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    global_mean = float(gray.mean())
    height, width = gray.shape
    worst = 0.0
    for y in range(0, height - block + 1, block):
        for x in range(0, width - block + 1, block):
            worst = max(
                worst, abs(float(gray[y : y + block, x : x + block].mean()) - global_mean)
            )
    return worst


def max_adjacent_mean_jump(img: Image.Image) -> tuple[float, float]:
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    columns = float(np.abs(np.diff(gray.mean(axis=0))).max())
    rows = float(np.abs(np.diff(gray.mean(axis=1))).max())
    return columns, rows


def laplacian_variance(img: Image.Image) -> float:
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    lap = (
        4.0 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    return float(np.abs(lap).var())


def mean_abs_diff(a: Image.Image | np.ndarray, b: Image.Image | np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.int16)
    bb = np.asarray(b, dtype=np.int16)
    return float(np.abs(aa - bb).mean())


# ---------------------------------------------------------------------------
# EngineInfo + factory
# ---------------------------------------------------------------------------
def test_engine_info_dataclass():
    info = engine_mod.EngineInfo("pil", "cpu")
    assert info.backend == "pil"
    assert info.device == "cpu"
    assert info.gpu is None
    assert info.model is None
    assert info.native_scale == 1
    assert info.models_available == ()
    expected = {
        "backend": "pil",
        "device": "cpu",
        "gpu": None,
        "model": None,
        "native_scale": 1,
        "models_available": [],
    }
    assert info.to_dict() == expected
    assert info.as_dict() == expected
    with pytest.raises(FrozenInstanceError):
        info.backend = "realesrgan"  # type: ignore[misc]


def test_pil_engine_info():
    engine = engine_mod.get_engine("pil")
    assert isinstance(engine, engine_mod.PILEngine)
    assert engine.info == engine_mod.EngineInfo("pil", "cpu", None, None, 1, ())
    assert engine.denoise_available is False


def test_get_engine_is_cached_per_mode():
    first = engine_mod.get_engine("pil")
    second = engine_mod.get_engine("pil")
    assert first is second


def test_get_engine_rejects_unknown_mode():
    with pytest.raises(engine_mod.EngineError):
        engine_mod.get_engine("bogus")


def test_engine_info_matches_active_engine():
    info = engine_mod.engine_info()
    assert isinstance(info, engine_mod.EngineInfo)
    assert info == engine_mod.get_engine().info
    assert info.backend in ("realesrgan", "pil")
    if info.backend == "realesrgan":
        assert info.models_available == tuple(models.available_model_ids())


def test_list_models_forwarded():
    assert engine_mod.list_models() == models.list_models()


def test_auto_falls_back_to_pil_when_realesrgan_unavailable(monkeypatch):
    def unavailable(**_kwargs):
        raise engine_mod.EngineError("模拟 Real-ESRGAN 不可用")

    monkeypatch.setattr(engine_mod, "_build_realesrgan", unavailable)
    engine = engine_mod.get_engine("auto")
    assert isinstance(engine, engine_mod.PILEngine)
    assert engine.info.backend == "pil"


def test_auto_uses_pil_when_no_models(monkeypatch):
    monkeypatch.setattr(engine_mod.models_mod, "available_model_ids", lambda: [])
    engine = engine_mod.get_engine("auto")
    assert isinstance(engine, engine_mod.PILEngine)


def test_forced_realesrgan_raises_without_models(monkeypatch):
    monkeypatch.setattr(engine_mod.models_mod, "available_model_ids", lambda: [])
    with pytest.raises(engine_mod.EngineError):
        engine_mod.get_engine("realesrgan")


def test_forced_realesrgan_raises_without_torch(monkeypatch):
    monkeypatch.setattr(engine_mod, "_import_torch", lambda: None)
    with pytest.raises(engine_mod.EngineError):
        engine_mod.get_engine("realesrgan")


# ---------------------------------------------------------------------------
# PIL backend: exact sizes, alpha, progress, cancellation, back-compat
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "src_w,src_h,out_w,out_h",
    [
        (100, 50, 200, 100),  # upscale
        (1000, 667, 1616, 1080),  # planned 1080p output of the odd sample
        (64, 64, 128, 128),
        (1000, 667, 500, 334),  # downscale
        (3840, 2160, 1920, 1080),  # 4K -> 1080p downscale
        (100, 50, 100, 50),  # same
        (37, 23, 211, 97),  # odd, aspect-changing target
    ],
)
def test_pil_upscale_exact_size(src_w, src_h, out_w, out_h):
    engine = engine_mod.get_engine("pil")
    out = engine.upscale(make_rgb(src_w, src_h), out_w, out_h)
    assert isinstance(out, Image.Image)
    assert out.size == (out_w, out_h)
    assert out.mode == "RGB"


def test_pil_same_size_pixels_unchanged():
    engine = engine_mod.get_engine("pil")
    src = make_rgb(44, 33)
    out = engine.upscale(src, 44, 33)
    assert out.size == (44, 33)
    assert out.mode == "RGB"
    assert out.tobytes() == src.convert("RGB").tobytes()


def test_pil_rgba_alpha_is_resized_and_merged():
    engine = engine_mod.get_engine("pil")
    src = make_rgba(100, 80)
    out = engine.upscale(src, 200, 160)
    assert out.size == (200, 160)
    assert out.mode == "RGBA"
    low, high = out.getchannel("A").getextrema()
    assert low < 40 and high > 215, (low, high)
    assert black_fraction(out.convert("RGB")) < 0.5


def test_pil_palette_with_transparency_becomes_rgba():
    engine = engine_mod.get_engine("pil")
    src = make_rgb(32, 24).convert("P")
    src.info["transparency"] = 0
    out = engine.upscale(src, 64, 48)
    assert out.size == (64, 48)
    assert out.mode == "RGBA"


@pytest.mark.parametrize("out_size", [(200, 100), (50, 25)])
def test_pil_progress_callback_is_monotonic_0_to_1(out_size):
    engine = engine_mod.get_engine("pil")
    values: list[float] = []
    out = engine.upscale(
        make_rgb(100, 50), out_size[0], out_size[1], progress_cb=values.append
    )
    assert out.size == out_size
    assert values
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 for v in values)
    assert values == sorted(values)


def test_pil_cancellation_propagates_untouched():
    engine = engine_mod.get_engine("pil")

    def cancel(_progress: float) -> None:
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        engine.upscale(make_rgb(100, 50), 200, 100, progress_cb=cancel)
    with pytest.raises(RuntimeError, match="cancelled"):
        engine.upscale(make_rgb(100, 50), 50, 25, progress_cb=cancel)


def test_pil_denoise_is_ignored_gracefully():
    engine = engine_mod.get_engine("pil")
    out = engine.upscale(make_rgb(80, 60), 160, 120, denoise=True, denoise_level=3)
    assert out.size == (160, 120)


def test_pil_tile_argument_is_ignored():
    engine = engine_mod.get_engine("pil")
    out = engine.upscale(make_rgb(80, 60), 160, 120, tile=999)
    assert out.size == (160, 120)


def test_pil_rejects_invalid_output_size():
    engine = engine_mod.get_engine("pil")
    for out_w, out_h in [(0, 10), (10, 0), (-1, 10), ("x", 10), (10, None)]:
        with pytest.raises(ValueError):
            engine.upscale(make_rgb(20, 20), out_w, out_h)


def test_pil_rejects_non_image_input():
    engine = engine_mod.get_engine("pil")
    with pytest.raises(TypeError):
        engine.upscale(np.zeros((10, 10, 3), dtype=np.uint8), 20, 20)


def test_pil_close_is_idempotent():
    engine = engine_mod.get_engine("pil")
    engine.close()
    engine.close()  # must not raise


def test_pil_backward_compatible_old_call_signature():
    """Old callers (pre-multi-model) pass no ``model=`` keyword."""
    engine = engine_mod.get_engine("pil")
    values: list[float] = []
    out = engine.upscale(
        make_rgb(120, 80), 240, 160, denoise=True, tile=256, progress_cb=values.append
    )
    assert out.size == (240, 160)
    assert values and values[-1] == pytest.approx(1.0)


def test_pil_resolve_model():
    engine = engine_mod.get_engine("pil")
    resolved = engine.resolve_model("auto", 1.5)
    assert resolved["id"] == "auto"
    assert resolved["family"] == "pil"
    assert resolved["used"] == "PIL (Lanczos)"
    assert resolved["native_scale"] == 1
    assert resolved["weight_file"] is None
    with pytest.raises(engine_mod.EngineError):
        engine.resolve_model("bogus-model", 1.5)


def test_pil_explicit_model_works_when_available():
    engine = engine_mod.get_engine("pil")
    out = engine.upscale(make_rgb(64, 48), 128, 96, model="realcugan", denoise_level=2)
    assert out.size == (128, 96)


def test_pil_works_without_any_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "no_models")
    engine = engine_mod.get_engine("pil")
    out = engine.upscale(make_rgb(50, 40), 100, 80)  # auto degrades to PIL
    assert out.size == (100, 80)
    with pytest.raises(engine_mod.EngineError):
        engine.resolve_model("realcugan", 1.5)  # explicit missing model -> 400
    with pytest.raises(engine_mod.EngineError):
        engine.upscale(make_rgb(50, 40), 100, 80, model="realesrgan-x4plus")


# ---------------------------------------------------------------------------
# Helper unit tests (no GPU needed)
# ---------------------------------------------------------------------------
def test_progress_guard_clamps_and_enforces_monotonicity():
    seen: list[float] = []
    guard = engine_mod._ProgressGuard(seen.append)
    for value in (-0.5, 0.25, 0.1, 1.5):
        guard(value)
    assert seen == [0.0, 0.25, 0.25, 1.0]
    assert guard.fired is False


def test_progress_guard_records_cancellation():
    def cancel(_progress: float) -> None:
        raise RuntimeError("cancelled")

    guard = engine_mod._ProgressGuard(cancel)
    with pytest.raises(RuntimeError, match="cancelled"):
        guard(0.1)
    assert guard.fired is True


def test_progress_guard_accepts_none():
    guard = engine_mod._ProgressGuard(None)
    guard(0.5)
    assert guard.fired is False


@pytest.mark.parametrize(
    "out_w,out_h", [(0, 10), (10, 0), (-1, 10), (10, -1), ("x", 10), (10, None)]
)
def test_validate_out_size_rejects_invalid(out_w, out_h):
    with pytest.raises(ValueError):
        engine_mod._validate_out_size(out_w, out_h)


def test_validate_out_size_accepts_positive_ints():
    assert engine_mod._validate_out_size(1920, 1080) == (1920, 1080)
    assert engine_mod._validate_out_size("1920", "1080") == (1920, 1080)


@pytest.mark.parametrize(
    "tile,expected",
    [(0, engine_mod.DEFAULT_TILE), (-5, engine_mod.DEFAULT_TILE),
     (1, engine_mod.MIN_TILE), (256, 256), (10_000, engine_mod.MAX_TILE),
     (None, engine_mod.DEFAULT_TILE)],
)
def test_normalise_tile(tile, expected):
    assert engine_mod._normalise_tile(tile) == expected


@pytest.mark.parametrize(
    "exc,is_oom",
    [
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), True),
        (MemoryError("out of memory"), True),
        (ValueError("bad shape"), False),
        (RuntimeError("cancelled"), False),
    ],
)
def test_is_oom_error(exc, is_oom):
    assert engine_mod._is_oom_error(exc) is is_oom


def test_oom_retry_halves_tile_until_success():
    calls: list[int] = []

    def run(tile: int) -> int:
        calls.append(tile)
        if len(calls) < 3:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return tile

    result = engine_mod._run_with_oom_retry(run, 256, empty_cache=lambda: None)
    assert calls == [256, 128, 64]
    assert result == 64


def test_oom_retry_gives_up_at_min_tile():
    calls: list[int] = []

    def run(tile: int) -> None:
        calls.append(tile)
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError):
        engine_mod._run_with_oom_retry(run, 256, empty_cache=lambda: None)
    assert calls == [256, 128, 64]


def test_oom_retry_respects_custom_min_tile():
    calls: list[int] = []

    def run(tile: int) -> None:
        calls.append(tile)
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError):
        engine_mod._run_with_oom_retry(run, 512, min_tile=32, empty_cache=lambda: None)
    assert calls == [512, 256, 128, 64, 32]


def test_oom_retry_does_not_swallow_other_errors():
    calls: list[int] = []

    def run(tile: int) -> None:
        calls.append(tile)
        raise ValueError("not an OOM")

    with pytest.raises(ValueError):
        engine_mod._run_with_oom_retry(run, 256, empty_cache=lambda: None)
    assert calls == [256]


@pytest.mark.parametrize(
    "model,denoise,level,expected_level",
    [
        ("auto", False, 0, 0),
        ("auto", True, 0, 0),  # rrdb: level stays 0, OpenCV flag is separate
        ("realesrgan-x4plus", True, 1, 1),
        ("realcugan", False, 0, 0),
        ("realcugan", False, 3, 3),
        ("realcugan", True, 0, 2),  # API.md §14.3
        ("realcugan", True, 1, 1),
        ("realcugan-conservative", True, 3, 0),  # fixed conservative weights
        ("realcugan", False, 9, 3),  # clamped
        ("realcugan", False, -1, 0),
    ],
)
def test_validate_model_and_level(model, denoise, level, expected_level):
    entry, effective = engine_mod._validate_model_and_level(model, denoise, level)
    assert entry["id"] == model
    assert effective == expected_level


def test_validate_model_and_level_rejects_unknown_model():
    with pytest.raises(engine_mod.EngineError):
        engine_mod._validate_model_and_level("nope", False, 0)


def test_display_name():
    rrdb = {"family": "rrdb", "weight_file": "RealESRGAN_x4plus_anime_6B.pth"}
    assert engine_mod._display_name(rrdb) == "RealESRGAN_x4plus_anime_6B"
    cugan = {"family": "cugan", "native_scale": 3, "variant": "denoise3x"}
    assert engine_mod._display_name(cugan) == "Real-CUGAN_3x_denoise3x"


# ---------------------------------------------------------------------------
# Engine model-resolution matrix (fake weights, no model loading)
# ---------------------------------------------------------------------------
@requires_weights
def test_engine_resolve_model_matrix(fake_models_dir):
    engine = engine_mod.RealESRGANEngine(torch_module=_torch(), device="cpu")
    contract_keys = {
        "id", "label", "used", "family", "native_scale", "weight_file", "denoise_level"
    }
    cases = [
        ("auto", 1.5, 0, "RealESRGAN_x2plus", "rrdb", 2, 0),
        ("auto", 3.0, 0, "RealESRGAN_x4plus", "rrdb", 4, 0),
        ("realesrgan-x4plus", 1.5, 0, "RealESRGAN_x4plus", "rrdb", 4, 0),
        ("realesrgan-x2plus", 4.0, 0, "RealESRGAN_x2plus", "rrdb", 2, 0),
        ("realesrgan-anime6b", 1.5, 0, "RealESRGAN_x4plus_anime_6B", "rrdb", 4, 0),
        ("realcugan", 1.5, 0, "Real-CUGAN_2x_no-denoise", "cugan", 2, 0),
        ("realcugan", 1.5, 2, "Real-CUGAN_2x_denoise2x", "cugan", 2, 2),
        ("realcugan", 2.5, 0, "Real-CUGAN_3x_no-denoise", "cugan", 3, 0),
        ("realcugan", 2.5, 3, "Real-CUGAN_3x_denoise3x", "cugan", 3, 3),
        ("realcugan", 4.0, 3, "Real-CUGAN_4x_denoise3x", "cugan", 4, 3),
        ("realcugan-conservative", 3.0, 3, "Real-CUGAN_3x_conservative", "cugan", 3, 0),
    ]
    for model, ratio, level, used, family, scale, effective in cases:
        resolved = engine.resolve_model(model, ratio, level)
        assert set(resolved) == contract_keys
        assert resolved["id"] == model
        assert resolved["used"] == used, (model, resolved)
        assert resolved["family"] == family
        assert resolved["native_scale"] == scale
        assert resolved["denoise_level"] == effective
        assert resolved["weight_file"]
    with pytest.raises(engine_mod.EngineError):
        engine.resolve_model("bogus", 1.5)
    engine.close()


def test_engine_resolve_auto_falls_back_when_preferred_missing(monkeypatch, tmp_path):
    if not TORCH_READY:
        pytest.skip("需要 torch")
    root = tmp_path / "models_x4_only"
    monkeypatch.setattr(config, "MODELS_DIR", root)
    path = root / "RealESRGAN_x4plus.pth"
    path.parent.mkdir(parents=True)
    with open(path, "wb") as handle:
        handle.truncate(models.MIN_WEIGHT_BYTES + 4096)
    engine = engine_mod.RealESRGANEngine(torch_module=_torch(), device="cpu")
    assert engine.resolve_model("auto", 1.5)["used"] == "RealESRGAN_x4plus"
    with pytest.raises(engine_mod.EngineError):
        engine.resolve_model("realesrgan-x2plus", 1.5)  # weights missing -> 400
    engine.close()


def test_engine_resolve_auto_falls_back_to_cugan(monkeypatch, tmp_path):
    if not TORCH_READY:
        pytest.skip("需要 torch")
    root = tmp_path / "models_cugan_only"
    monkeypatch.setattr(config, "MODELS_DIR", root)
    path = root / "realcugan" / "up2x-latest-no-denoise.pth"
    path.parent.mkdir(parents=True)
    with open(path, "wb") as handle:
        handle.truncate(models.MIN_WEIGHT_BYTES + 4096)
    engine = engine_mod.RealESRGANEngine(torch_module=_torch(), device="cpu")
    resolved = engine.resolve_model("auto", 1.5)
    assert resolved["id"] == "auto"
    assert "Real-CUGAN_2x" in resolved["used"]
    assert engine.info.model == "Real-CUGAN"
    engine.close()


@pytest.mark.parametrize(
    "files,expected_model,expected_scale",
    [
        (["RealESRGAN_x4plus.pth"], "RealESRGAN_x4plus", 4),
        (["RealESRGAN_x2plus.pth"], "RealESRGAN_x2plus", 2),
        (["RealESRGAN_x4plus_anime_6B.pth"], "RealESRGAN_x4plus_anime_6B", 4),
        (["realcugan/up2x-latest-no-denoise.pth"], "Real-CUGAN", 2),
    ],
)
def test_engine_primary_display_model(monkeypatch, tmp_path, files, expected_model,
                                     expected_scale):
    if not TORCH_READY:
        pytest.skip("需要 torch")
    root = tmp_path / "models_primary"
    monkeypatch.setattr(config, "MODELS_DIR", root)
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            handle.truncate(models.MIN_WEIGHT_BYTES + 4096)
    engine = engine_mod.RealESRGANEngine(torch_module=_torch(), device="cpu")
    assert engine.info.model == expected_model
    assert engine.info.native_scale == expected_scale
    assert engine.info.models_available
    engine.close()


def test_engine_denoise_available_matches_cv2(fake_models_dir):
    if not TORCH_READY:
        pytest.skip("需要 torch")
    engine = engine_mod.RealESRGANEngine(torch_module=_torch(), device="cpu")
    assert engine.denoise_available is (engine_mod._import_cv2() is not None)
    engine.close()


# ---------------------------------------------------------------------------
# RealESRGAN engine: downscale must never run the network
# ---------------------------------------------------------------------------
@requires_weights
def test_downscale_and_same_never_run_network(monkeypatch):
    engine = engine_mod.RealESRGANEngine(device="cpu")
    try:
        def explode(*_args, **_kwargs):
            raise AssertionError("downscale/same 不允许跑 SR 网络")

        monkeypatch.setattr(engine, "_run_sr", explode)
        src = make_rgb(320, 240)
        for out_w, out_h in [(160, 120), (320, 240)]:
            out = engine.upscale(src, out_w, out_h)
            assert out.size == (out_w, out_h)
            assert out.mode == "RGB"
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# Vendored networks: strict checkpoint loading + forward shapes
# ---------------------------------------------------------------------------
@requires_model("realesrgan-x4plus")
def test_rrdbnet_checkpoints_load_strict():
    torch = _torch()
    from app.rrdbnet import RRDBNet, load_rrdbnet

    x4 = load_rrdbnet(MODEL_X4, 4, "cpu")
    assert isinstance(x4, RRDBNet)
    assert x4.scale == 4
    assert not x4.training
    assert x4.conv_first.in_channels == 3
    assert x4.conv_first.out_channels == 64
    assert len(x4.body) == 23
    assert sum(p.numel() for p in x4.parameters()) == 16_697_987
    assert not any(p.requires_grad for p in x4.parameters())
    checkpoint = torch.load(MODEL_X4, map_location="cpu", weights_only=True)
    assert set(x4.state_dict()) == set(checkpoint["params_ema"])
    with torch.inference_mode():
        out4 = x4(torch.zeros(1, 3, 8, 8))
    assert tuple(out4.shape) == (1, 3, 32, 32)
    assert bool(torch.isfinite(out4).all())


@requires_model("realesrgan-x2plus")
def test_rrdbnet_x2_weight_strict():
    torch = _torch()
    from app.rrdbnet import load_rrdbnet

    x2 = load_rrdbnet(MODEL_X2, 2, "cpu")
    assert x2.conv_first.in_channels == 12  # pixel_unshuffle twice
    checkpoint = torch.load(MODEL_X2, map_location="cpu", weights_only=True)
    assert set(x2.state_dict()) == set(checkpoint["params_ema"])
    with torch.inference_mode():
        out2 = x2(torch.zeros(1, 3, 8, 8))
    assert tuple(out2.shape) == (1, 3, 16, 16)


@requires_model("realesrgan-anime6b")
def test_anime6b_uses_six_blocks():
    torch = _torch()
    from app.rrdbnet import load_rrdbnet

    anime = load_rrdbnet(MODEL_ANIME, 4, "cpu", num_block=6)
    assert len(anime.body) == 6
    x4 = load_rrdbnet(MODEL_X4, 4, "cpu")
    assert sum(p.numel() for p in anime.parameters()) < sum(
        p.numel() for p in x4.parameters()
    )
    with torch.inference_mode():
        out = anime(torch.zeros(1, 3, 8, 8))
    assert tuple(out.shape) == (1, 3, 32, 32)


def test_load_rrdbnet_error_handling(tmp_path):
    if _torch() is None:
        pytest.skip("需要 torch")
    from app.rrdbnet import load_rrdbnet

    with pytest.raises(FileNotFoundError):
        load_rrdbnet(tmp_path / "missing.pth", 4, "cpu")
    dummy = tmp_path / "dummy.pth"
    dummy.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError):
        load_rrdbnet(dummy, 3, "cpu")  # unsupported scale, checked before loading


@requires_model("realesrgan-x4plus")
def test_extract_state_dict_variants():
    torch = _torch()
    from app.rrdbnet import extract_state_dict

    weights = {"conv_first.weight": torch.zeros(1)}
    assert dict(extract_state_dict({"params_ema": weights})) == weights
    assert dict(extract_state_dict({"params": weights})) == weights
    assert dict(extract_state_dict(weights)) == weights
    stripped = extract_state_dict({"params_ema": {"module.conv_first.weight": torch.zeros(1)}})
    assert "conv_first.weight" in stripped
    with pytest.raises(ValueError):
        extract_state_dict({"something": 1})


# ---------------------------------------------------------------------------
# Vendored Real-CUGAN
# ---------------------------------------------------------------------------
def test_cugan_module_structure():
    from app import cugan

    assert cugan.CUGAN_CLASSES == {2: cugan.UpCunet2x, 3: cugan.UpCunet3x,
                                   4: cugan.UpCunet4x}
    for name in ("SEBlock", "UNetConv", "UNet1", "UNet1x3", "UNet2", "q", "dq",
                 "load_cugan"):
        assert hasattr(cugan, name), name
    assert not hasattr(cugan, "RealWaifuUpScaler")  # application helper dropped
    source = Path(cugan.__file__).read_text(encoding="utf-8")
    assert "Copyright (c) 2022 bilibili" in source
    assert "MIT License" in source
    assert "import basicsr" not in source
    assert "import realcugan" not in source


@pytest.mark.skipif(not CUGAN_DIR.is_dir(), reason="缺少 models/realcugan 权重")
def test_cugan_all_weights_load_strict():
    import torch
    from app.cugan import CUGAN_CLASSES, load_cugan

    files = sorted(CUGAN_DIR.glob("up*x-latest-*.pth"))
    if not files:
        pytest.skip("models/realcugan 为空")
    assert len(files) == 11, [path.name for path in files]
    forward_reps = {
        "up2x-latest-no-denoise.pth",
        "up3x-latest-denoise3x.pth",
        "up4x-latest-conservative.pth",
    }
    for path in files:
        scale = int(path.name.split("x")[0].replace("up", ""))
        model = load_cugan(scale, path, "cpu")
        assert isinstance(model, CUGAN_CLASSES[scale])
        assert model.training is False
        assert not any(p.requires_grad for p in model.parameters())
        if path.name in forward_reps:
            with torch.inference_mode():
                out = model(torch.zeros(1, 3, 32, 32), 0, 0, 1.0, False)
            assert out.dtype == torch.uint8  # CUGAN v3 contract: uint8 [0,255]
            assert tuple(out.shape) == (1, 3, 32 * scale, 32 * scale)
            assert int(out.max()) <= 255 and int(out.min()) >= 0


def test_load_cugan_errors(tmp_path):
    if _torch() is None:
        pytest.skip("需要 torch")
    from app.cugan import load_cugan

    with pytest.raises(FileNotFoundError):
        load_cugan(2, tmp_path / "missing.pth", "cpu")
    with pytest.raises(ValueError):
        load_cugan(5, tmp_path / "missing.pth", "cpu")


# ---------------------------------------------------------------------------
# GPU: tiling regression tests (would fail on the old "black blocks" bug)
# ---------------------------------------------------------------------------
@requires_gpu_model("auto")
def test_multi_tile_mosaic_x2_has_no_black_blocks():
    engine = engine_mod.get_engine()
    assert engine.info.backend == "realesrgan"
    assert engine_mod.get_engine("realesrgan") is engine  # cached singleton

    src = make_gradient(700, 700)  # 3x3 tiles at tile=256 -> multi-tile mosaic
    out = engine.upscale(src, 1400, 1400, tile=256)

    assert out.size == (1400, 1400)
    assert any("x2plus" in key for key in engine._modules), engine._modules.keys()
    assert black_fraction(out) < 0.005, f"黑块占比过高：{black_fraction(out):.3f}"
    assert max_block_mean_deviation(out) < 5.0
    jump_cols, jump_rows = max_adjacent_mean_jump(out)
    assert jump_cols < 3.0 and jump_rows < 3.0
    assert abs(float(np.asarray(out).mean()) - float(np.asarray(src).mean())) < 8.0


@requires_gpu_model("realesrgan-x4plus")
def test_multi_tile_mosaic_x4_has_no_black_blocks():
    engine = engine_mod.get_engine()
    src = make_gradient(700, 700)
    out = engine.upscale(src, 2100, 2100, model="realesrgan-x4plus", tile=256)

    assert out.size == (2100, 2100)
    assert any("x4plus.pth" in key for key in engine._modules)
    assert black_fraction(out) < 0.005, f"黑块占比过高：{black_fraction(out):.3f}"
    assert max_block_mean_deviation(out) < 5.0
    jump_cols, jump_rows = max_adjacent_mean_jump(out)
    assert jump_cols < 3.0 and jump_rows < 3.0
    assert abs(float(np.asarray(out).mean()) - float(np.asarray(src).mean())) < 8.0


# ---------------------------------------------------------------------------
# GPU: full model matrix on a small image + model-specific behaviour
# ---------------------------------------------------------------------------
@requires_gpu
def test_engine_model_matrix_small_gpu():
    engine = engine_mod.get_engine()
    src = Image.open(ROOT / "samples" / "photo_1280x720.jpg").convert("RGB") \
        if (ROOT / "samples" / "photo_1280x720.jpg").is_file() else make_photo_like()
    src = src.resize((320, 240), LANCZOS)
    target = (480, 360)  # ratio 1.5

    outputs: dict[str, np.ndarray] = {}
    cases = [
        ("auto", {}),
        ("realesrgan-x4plus", {}),
        ("realesrgan-x2plus", {}),
        ("realesrgan-anime6b", {}),
        ("realcugan", {"denoise_level": 0}),
        ("realcugan", {"denoise_level": 3}),
        ("realcugan-conservative", {}),
    ]
    for model_id, kwargs in cases:
        if not _model_ready(model_id):
            pytest.skip(f"模型 {model_id} 权重缺失")
        label = model_id + (f"+lvl{kwargs['denoise_level']}" if kwargs else "")
        resolved = engine.resolve_model(model_id, 1.5, kwargs.get("denoise_level", 0))
        out = engine.upscale(src, *target, model=model_id, **kwargs)
        assert out.size == target, label
        assert black_fraction(out) < 0.005, label
        assert float(np.asarray(out).std()) > 3.0, label
        outputs[label] = np.asarray(out)
        print(f"[模型矩阵] {label:<28} used={resolved['used']:<30} "
              f"mean={outputs[label].mean():.1f} std={outputs[label].std():.1f}")

    if "realesrgan-x4plus" in outputs and "realesrgan-anime6b" in outputs:
        assert mean_abs_diff(outputs["realesrgan-x4plus"],
                             outputs["realesrgan-anime6b"]) > 1.0
    if "realcugan+lvl0" in outputs and "realcugan+lvl3" in outputs:
        diff = mean_abs_diff(outputs["realcugan+lvl0"], outputs["realcugan+lvl3"])
        print(f"[模型矩阵] cugan lvl0 vs lvl3 mean|diff| = {diff:.2f}")
        assert diff > 1.0
    if "realcugan+lvl0" in outputs and "realcugan-conservative" in outputs:
        assert mean_abs_diff(outputs["realcugan+lvl0"],
                             outputs["realcugan-conservative"]) > 0.5

    # uint8 pipeline sanity: CUGAN output must not be greyed out / overflowed.
    cugan = outputs.get("realcugan+lvl0")
    if cugan is not None:
        src_mean = float(np.asarray(src).mean())
        assert abs(float(cugan.mean()) - src_mean) < 15.0
        assert cugan.max() <= 255 and cugan.min() >= 0


@requires_gpu_model("realcugan")
def test_cugan_denoise_true_maps_to_level2():
    engine = engine_mod.get_engine()
    src = make_noisy(240, 160)
    out = engine.upscale(src, 480, 320, model="realcugan", denoise=True, denoise_level=0)
    assert out.size == (480, 320)
    # API.md §14.3: denoise=true without a level == native level 2.
    assert any("up2x-latest-denoise2x" in key for key in engine._modules), engine._modules


@requires_gpu_model("realesrgan-x4plus")
def test_rrdb_opencv_denoise_changes_output():
    engine = engine_mod.get_engine()
    if not engine.denoise_available:
        pytest.skip("未安装 OpenCV，denoise 会被忽略")
    src = make_noisy(320, 240)
    plain = engine.upscale(src, 640, 480, model="realesrgan-x4plus", denoise=False)
    denoised = engine.upscale(src, 640, 480, model="realesrgan-x4plus", denoise=True)
    assert plain.size == denoised.size == (640, 480)
    diff = mean_abs_diff(plain, denoised)
    print(f"[denoise] rrdb OpenCV pre-denoise mean|diff| = {diff:.2f}")
    assert diff > 1.0


@requires_gpu_model("auto")
def test_cancellation_propagates_from_gpu_engine():
    engine = engine_mod.get_engine()

    def cancel(_progress: float) -> None:
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        engine.upscale(make_rgb(320, 240), 480, 360, progress_cb=cancel)


# ---------------------------------------------------------------------------
# GPU: deadlines / matrix-level acceptance
# ---------------------------------------------------------------------------
@requires_gpu_model("auto")
def test_upscale_720p_to_1080p_meets_deadline():
    torch = _torch()
    engine = engine_mod.get_engine()
    src, real_photo = sample_photo()

    progress: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    out = engine.upscale(src, 1920, 1080, progress_cb=progress.append)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
    print(f"\n[GPU] auto 1280x720 -> 1920x1080: {elapsed:.2f}s, peak {peak_mb:.1f} MB")

    assert out.size == (1920, 1080)
    assert out.mode == "RGB"
    assert elapsed <= 45.0, f"720p->1080p 需 ≤45s（auto/x2plus），实际 {elapsed:.1f}s"
    assert progress and progress[-1] == pytest.approx(1.0)
    assert progress == sorted(progress)
    assert black_fraction(out) < 0.01
    if real_photo:
        baseline = src.resize((1920, 1080), LANCZOS)
        assert laplacian_variance(out) > laplacian_variance(baseline) * 1.3


@requires_gpu_model("realcugan")
def test_realcugan_720p_to_1080p_deadline():
    torch = _torch()
    engine = engine_mod.get_engine()
    src, _ = sample_photo()
    resolved = engine.resolve_model("realcugan", 1.5, 2)
    assert resolved["used"] == "Real-CUGAN_2x_denoise2x"

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    out = engine.upscale(src, 1920, 1080, model="realcugan", denoise_level=2)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
    print(f"[GPU] realcugan 1280x720 -> 1920x1080: {elapsed:.2f}s, peak {peak_mb:.1f} MB")

    assert out.size == (1920, 1080)
    assert elapsed <= 90.0, f"其他模型 720p->1080p 需 ≤90s，实际 {elapsed:.1f}s"
    assert black_fraction(out) < 0.005
    assert float(np.asarray(out).std()) > 5.0
    assert abs(float(np.asarray(out).mean()) - float(np.asarray(src).mean())) < 20.0


@requires_gpu_model("realcugan")
def test_realcugan_720p_to_4k_uses_3x():
    engine = engine_mod.get_engine()
    torch = _torch()
    src, _ = sample_photo()
    resolved = engine.resolve_model("realcugan", 3.0, 0)
    assert resolved["native_scale"] == 3

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    out = engine.upscale(src, 3840, 2160, model="realcugan")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
    print(f"[GPU] realcugan(3x) 1280x720 -> 3840x2160: {elapsed:.2f}s, "
          f"peak {peak_mb:.1f} MB")

    assert out.size == (3840, 2160)
    assert elapsed <= 180.0
    assert black_fraction(out) < 0.005
    assert any("up3x" in key for key in engine._modules)


@requires_gpu_model("auto")
def test_auto_1080p_to_4k():
    torch = _torch()
    engine = engine_mod.get_engine()
    sample = ROOT / "samples" / "photo_beach_1920x1200.jpg"
    if sample.is_file():
        src = Image.open(sample).convert("RGB")
    else:
        src = make_photo_like(1920, 1200)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    out = engine.upscale(src, 3840, 2400)  # ratio 2.0 -> x2plus
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(f"[GPU] auto 1920x1200 -> 3840x2400: {elapsed:.2f}s, "
          f"peak {torch.cuda.max_memory_allocated() / 2 ** 20:.1f} MB")
    assert out.size == (3840, 2400)
    assert elapsed <= 180.0
    assert black_fraction(out) < 0.005


@requires_gpu_model("auto")
def test_x4_model_path_and_tile_override():
    engine = engine_mod.get_engine()
    src = make_rgb(320, 180)
    out = engine.upscale(src, 1280, 720, tile=128)  # ratio 4.0 -> x4plus
    assert out.size == (1280, 720)
    assert black_fraction(out) < 0.005
    assert float(np.asarray(out).std()) > 5.0


@requires_gpu_model("realcugan")
def test_cugan_tiny_image_is_valid():
    """Very small images must still work (patches are edge-padded up to 32 px)."""
    engine = engine_mod.get_engine()
    out = engine.upscale(make_rgb(16, 16), 32, 32, model="realcugan")
    assert out.size == (32, 32)
    assert black_fraction(out) < 0.05


@requires_gpu_model("realcugan")
def test_cugan_rgba_upscale_preserves_alpha():
    engine = engine_mod.get_engine()
    out = engine.upscale(make_rgba(128, 96), 192, 144, model="realcugan")
    assert out.size == (192, 144)
    assert out.mode == "RGBA"
    low, high = out.getchannel("A").getextrema()
    assert low < 40 and high > 215, (low, high)
    assert black_fraction(out.convert("RGB")) < 0.05
