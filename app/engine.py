"""Super-resolution inference engine for PicPolish (multi-model).

Backends and models (CONTRACT §3.4 / API.md §14):

``realesrgan``
    PyTorch inference.  Three vendored RRDB networks are supported:
    ``RealESRGAN_x4plus`` (num_block=23, x4), ``RealESRGAN_x2plus`` (x2) and
    ``RealESRGAN_x4plus_anime_6B`` (num_block=6, x4).  ``auto`` picks x2plus for
    upscale ratios <= 2 and x4plus otherwise.

``realesrgan`` + Real-CUGAN
    The Real-CUGAN v3 families (``UpCunet2x/3x/4x``, vendored in
    :mod:`app.cugan`) are selected by ratio (<=2 -> 2x, <=3 -> 3x, else 4x) and
    expose native denoise levels (``no-denoise`` / ``denoise1x/2x/3x``).
    Their network output is **uint8 [0,255]** and is converted to PIL directly.

``pil``
    Dependency-free fallback: LANCZOS resize + gentle ``UnsharpMask`` for
    upscaling (downscaling never sharpens).  Used automatically whenever torch,
    CUDA or the model weights are unavailable -- it never raises.

Inference details:
* ``torch.inference_mode()`` + external tiling (default ``tile=256``, RRDB
  context pad 10 px, Real-CUGAN context pad 32 px) with CUDA-OOM retry that
  halves the tile down to 64 px;
* the SR output is then resized with ``Image.LANCZOS`` to the **exact** target
  size (plus a mild unsharp mask); downscale/same-size targets skip the network;
* RGBA inputs keep their alpha channel (resized separately with LANCZOS);
* ``denoise=True`` runs OpenCV ``fastNlMeansDenoisingColored`` before RRDB
  inference (``denoise_available`` reports whether cv2 is installed); Real-CUGAN
  uses its native denoise weights instead;
* ``progress_cb`` is called with 0..1 while tiles are processed and any
  exception it raises (e.g. ``RuntimeError("cancelled")``) propagates untouched;
* ``torch.cuda.empty_cache()`` runs after every request.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, ImageFilter

from . import config
from . import models as models_mod

__all__ = [
    "EngineInfo",
    "EngineError",
    "BaseEngine",
    "PILEngine",
    "RealESRGANEngine",
    "get_engine",
    "engine_info",
    "list_models",
    "shutdown_engines",
]

logger = logging.getLogger(__name__)

ProgressCB = Callable[[float], None]

# PIL keeps the enum behind Image.Resampling since Pillow 9; Image.LANCZOS is a
# compatibility alias that still works on Pillow 12.
_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

#: Overlap (in source pixels) kept around every RRDB tile so seams are invisible.
TILE_OVERLAP = 10
#: Real-CUGAN needs more context (U-Net + SE blocks); CONTRACT §3.4: pad >= 32.
CUGAN_TILE_OVERLAP = 32
#: Real-CUGAN's own reflect padding (14/18/19 px) requires a minimal spatial
#: size; tiny edge patches (very small images) are edge-padded up to this size.
CUGAN_MIN_INPUT = 32
#: Inference tile size used when the caller passes ``tile=0``.
DEFAULT_TILE = int(getattr(config, "TILE", 256) or 256)
#: Smallest tile the OOM retry loop will shrink to.
MIN_TILE = 64
#: Upper sanity bound for a caller supplied tile size.
MAX_TILE = 2048

#: Gentle sharpening applied after an SR output is resized to the exact target.
SR_UNSHARP = {"radius": 1.0, "percent": 28, "threshold": 3}
#: Sharpening used by the PIL fallback for upscales (CONTRACT §3.4).
PIL_UNSHARP = {"radius": 1.2, "percent": 45, "threshold": 3}

_ENGINE_MODES = ("auto", "realesrgan", "pil")
#: pixel_unshuffle inside RRDBNet requires the tile size to be divisible by this.
_UNSHUFFLE_ALIGN = {1: 4, 2: 2, 4: 1}
#: Preferred model order when reporting the primary display model.
_PRIMARY_MODEL_ORDER = (
    ("realesrgan-x4plus", "RealESRGAN_x4plus", 4),
    ("realesrgan-x2plus", "RealESRGAN_x2plus", 2),
    ("realesrgan-anime6b", "RealESRGAN_x4plus_anime_6B", 4),
    ("realcugan", "Real-CUGAN", 2),
    ("realcugan-conservative", "Real-CUGAN", 2),
)
#: Candidates used when ``model="auto"`` has to degrade to an available model.
_AUTO_FALLBACK_ORDER = (
    "realesrgan-x2plus",
    "realesrgan-x4plus",
    "realesrgan-anime6b",
    "realcugan",
    "realcugan-conservative",
)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EngineInfo:
    """Immutable description of the active engine (serialisable for /api/health)."""

    backend: str  # "realesrgan" | "pil"
    device: str  # "cuda" | "cpu"
    gpu: Optional[str] = None
    model: Optional[str] = None  # primary display name, e.g. "RealESRGAN_x4plus"
    native_scale: int = 1  # 4 / 2 / 1
    models_available: tuple[str, ...] = ()  # ready model ids (registry order)

    def to_dict(self) -> dict:
        """Return the JSON-ready mapping used by the API layer."""
        return {
            "backend": self.backend,
            "device": self.device,
            "gpu": self.gpu,
            "model": self.model,
            "native_scale": int(self.native_scale),
            "models_available": list(self.models_available),
        }

    #: Alias, so both ``info.to_dict()`` and ``info.as_dict()`` work.
    as_dict = to_dict


class EngineError(RuntimeError):
    """Raised for explicit engine/model failures (API maps it to 400/503)."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
class _ProgressGuard:
    """Clamp progress values, enforce monotonicity and detect cancellation.

    If ``progress_cb`` raises (this is how ``jobs.py`` cancels a running job),
    the exception is remembered via :attr:`fired` and re-raised, so the engine
    never mistakes a cancellation for a recoverable inference failure.
    """

    __slots__ = ("_cb", "fired", "_last")

    def __init__(self, cb: Optional[ProgressCB]):
        self._cb = cb
        self.fired = False
        self._last = -1.0

    def __call__(self, value: float) -> None:
        if self._cb is None:
            return
        value = 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))
        if value < self._last:
            value = self._last
        self._last = value
        try:
            self._cb(value)
        except BaseException:
            self.fired = True
            raise


def _has_alpha(img: Image.Image) -> bool:
    return "A" in img.getbands() or "transparency" in img.info


def _split_alpha(img: Image.Image) -> tuple[Image.Image, Optional[Image.Image]]:
    """Return ``(rgb, alpha_or_None)``; converts palette/LA/CMYK safely."""
    if _has_alpha(img):
        rgba = img.convert("RGBA")
        return rgba.convert("RGB"), rgba.getchannel("A")
    return img.convert("RGB"), None


def _merge_alpha(
    rgb: Image.Image, alpha: Optional[Image.Image], out_w: int, out_h: int
) -> Image.Image:
    """Re-attach a separately resized alpha channel (LANCZOS) to ``rgb``."""
    if alpha is None:
        return rgb
    if alpha.size != (out_w, out_h):
        alpha = alpha.resize((out_w, out_h), _LANCZOS)
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def _resize_rgb(
    rgb: Image.Image, out_w: int, out_h: int, *, unsharp: Optional[dict] = None
) -> Image.Image:
    """High-quality LANCZOS resize to the exact target, optional mild sharpening."""
    if rgb.size == (out_w, out_h):
        return rgb.copy()
    out = rgb.resize((out_w, out_h), _LANCZOS)
    if unsharp:
        out = out.filter(ImageFilter.UnsharpMask(**unsharp))
    return out


def _validate_out_size(out_w: int, out_h: int) -> tuple[int, int]:
    try:
        width = int(out_w)
        height = int(out_h)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"输出尺寸必须是正整数：{out_w!r}x{out_h!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"输出尺寸必须是正整数：{out_w!r}x{out_h!r}")
    return width, height


def _normalise_tile(tile: int) -> int:
    try:
        value = int(tile)
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        value = DEFAULT_TILE
    return max(MIN_TILE, min(value, MAX_TILE))


def _is_oom_error(exc: BaseException) -> bool:
    """True for CUDA/host out-of-memory errors (retryable by shrinking tiles)."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    message = str(exc).lower()
    return "out of memory" in message or "显存不足" in message


def _run_with_oom_retry(
    fn: Callable[[int], object],
    tile: int,
    *,
    min_tile: int = MIN_TILE,
    empty_cache: Optional[Callable[[], None]] = None,
):
    """Call ``fn(tile)``; on CUDA OOM halve ``tile`` (down to ``min_tile``) and retry.

    ``fn`` fills the pre-allocated output mosaic.  Non-OOM exceptions (e.g.
    cancellation raised by ``progress_cb``) propagate unchanged.
    """
    current = max(int(min_tile), int(tile))
    while True:
        try:
            return fn(current)
        except Exception as exc:  # noqa: BLE001 - OOM detection below
            if not _is_oom_error(exc) or current <= min_tile:
                raise
            new_tile = max(int(min_tile), current // 2)
            logger.warning(
                "GPU 显存不足（%s），推理 tile %d → %d 重试",
                str(exc).splitlines()[0][:160], current, new_tile,
            )
            if empty_cache is not None:
                empty_cache()
            else:  # pragma: no cover - default path exercised on a real GPU
                _empty_cuda_cache()
            current = new_tile


def _import_torch():
    """Import torch lazily so the PIL backend never pays for it."""
    try:
        import torch  # noqa: PLC0415 - intentional lazy import

        return torch
    except Exception as exc:  # pragma: no cover - depends on the environment
        logger.debug("PyTorch 不可用：%s", exc)
        return None


def _import_cv2():
    try:
        import cv2  # noqa: PLC0415 - optional dependency

        return cv2
    except Exception:
        return None


def _cuda_available(torch) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - broken CUDA install
        return False


def _empty_cuda_cache() -> None:
    torch = _import_torch()
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - defensive
        pass


# ---------------------------------------------------------------------------
# Model resolution helpers
# ---------------------------------------------------------------------------
def _validate_model_and_level(
    model: str, denoise: bool, denoise_level: int
) -> tuple[dict, int]:
    """Validate the requested model id and normalise the denoise level.

    * unknown ids raise :class:`EngineError` (API 400);
    * Real-CUGAN ``denoise=True`` without an explicit level means level 2
      (API.md §14.3);
    * ``realcugan-conservative`` always uses its fixed weights (level 0).
    """
    try:
        entry = models_mod.get_model(model)
    except models_mod.InvalidModelError as exc:
        raise EngineError(str(exc)) from exc
    try:
        level = int(denoise_level)
    except (TypeError, ValueError):
        level = 0
    level = max(0, min(3, level))
    if entry["family"] == "cugan":
        if model == "realcugan-conservative":
            level = 0
        elif model == "realcugan" and denoise and level == 0:
            level = 2  # API.md §14.3: denoise=true without level == level 2
    return entry, level


def _display_name(resolved: dict) -> str:
    """Human readable identifier of the concrete checkpoint (``model_used``)."""
    if resolved.get("family") == "rrdb":
        return Path(str(resolved["weight_file"])).stem
    variant = str(resolved.get("variant") or "no-denoise")
    return f"Real-CUGAN_{int(resolved['native_scale'])}x_{variant}"


def _pil_resolution(model: str, entry: dict) -> dict:
    return {
        "id": model,
        "label": entry.get("label") or "兼容模式",
        "used": "PIL (Lanczos)",
        "family": "pil",
        "native_scale": 1,
        "weight_file": None,
        "denoise_level": 0,
    }


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
class BaseEngine:
    """Common interface for every engine backend."""

    info: EngineInfo
    #: Whether ``denoise=True`` can actually do something (needs OpenCV).
    denoise_available: bool = False

    def upscale(
        self,
        img: Image.Image,
        out_w: int,
        out_h: int,
        *,
        model: str = "auto",
        denoise: bool = False,
        denoise_level: int = 0,
        tile: int = 0,
        progress_cb: Optional[ProgressCB] = None,
    ) -> Image.Image:
        """Upscale/downscale ``img`` to exactly ``(out_w, out_h)`` pixels."""
        raise NotImplementedError

    def resolve_model(self, model: str, ratio: float, denoise_level: int = 0) -> dict:
        """Describe which concrete checkpoint a request would use (no loading)."""
        raise NotImplementedError

    def close(self) -> None:
        """Release device memory / model references (idempotent)."""
        return None


class PILEngine(BaseEngine):
    """Fallback engine: LANCZOS + light ``UnsharpMask``, no torch required."""

    def __init__(self) -> None:
        self.info = EngineInfo(
            backend="pil", device="cpu", gpu=None, model=None, native_scale=1,
            models_available=(),
        )
        self.denoise_available = False

    def resolve_model(self, model: str, ratio: float, denoise_level: int = 0) -> dict:
        entry, _level = _validate_model_and_level(model, False, denoise_level)
        if model != models_mod.DEFAULT_MODEL and not entry.get("available"):
            raise EngineError(
                f"模型 {model} 的权重缺失，请运行 scripts/fetch_models.py 下载"
            )
        return _pil_resolution(model, entry)

    def upscale(
        self,
        img: Image.Image,
        out_w: int,
        out_h: int,
        *,
        model: str = "auto",
        denoise: bool = False,
        denoise_level: int = 0,
        tile: int = 0,
        progress_cb: Optional[ProgressCB] = None,
    ) -> Image.Image:
        if not isinstance(img, Image.Image):  # pragma: no cover - defensive
            raise TypeError("img 必须是 PIL.Image.Image")
        out_w, out_h = _validate_out_size(out_w, out_h)
        self.resolve_model(model, 1.0, denoise_level)  # validate id/availability
        cb = _ProgressGuard(progress_cb)

        src_rgb, alpha = _split_alpha(img)
        src_w, src_h = src_rgb.size
        same = (src_w, src_h) == (out_w, out_h)
        upscaling = out_w * out_h > src_w * src_h

        cb(0.0)
        if same:
            rgb = src_rgb.copy()
            cb(0.6)
        else:
            rgb = _resize_rgb(
                src_rgb, out_w, out_h, unsharp=PIL_UNSHARP if upscaling else None
            )
            cb(0.8)
        result = _merge_alpha(rgb, alpha, out_w, out_h)
        cb(1.0)
        return result


class RealESRGANEngine(BaseEngine):
    """PyTorch engine: Real-ESRGAN (RRDB) + Real-CUGAN (U-Net) checkpoints."""

    def __init__(self, *, torch_module=None, device: Optional[str] = None) -> None:
        torch = torch_module if torch_module is not None else _import_torch()
        if torch is None:
            raise EngineError("PyTorch 不可用，无法使用 Real-ESRGAN 引擎")
        self._torch = torch
        self._lock = threading.RLock()
        self._modules: dict[str, object] = {}

        available = list(models_mod.available_model_ids())
        if not [mid for mid in available if mid != models_mod.DEFAULT_MODEL]:
            raise EngineError(
                "未找到任何超分模型权重（models/ 为空），请运行 scripts/fetch_models.py 下载"
            )
        self._available = available

        if device is None:
            device = "cuda" if _cuda_available(torch) else "cpu"
        self._device = torch.device(device)

        primary_model, primary_scale = self._pick_primary(available)
        gpu_name = None
        if self._device.type == "cuda":  # pragma: no cover - needs a GPU
            gpu_name = self._gpu_name(torch)

        self.info = EngineInfo(
            backend="realesrgan",
            device=self._device.type,
            gpu=gpu_name,
            model=primary_model,
            native_scale=primary_scale,
            models_available=tuple(available),
        )
        self.denoise_available = _import_cv2() is not None
        logger.info(
            "多模型引擎就绪：device=%s gpu=%s primary=%s denoise=%s models=%s",
            self.info.device, self.info.gpu, self.info.model,
            self.denoise_available, available,
        )

    # -- model management ---------------------------------------------------
    @staticmethod
    def _pick_primary(available: list[str]) -> tuple[Optional[str], int]:
        for model_id, display, scale in _PRIMARY_MODEL_ORDER:
            if model_id in available:
                return display, scale
        return None, 1

    def _gpu_name(self, torch) -> Optional[str]:  # pragma: no cover - needs a GPU
        try:
            return torch.cuda.get_device_name(self._device)
        except Exception:
            try:
                return torch.cuda.get_device_name(0)
            except Exception:
                return None

    def _resolve_weights(self, model: str, ratio: float, level: int) -> dict:
        """Resolve to a checkpoint; ``auto`` degrades to any available model."""
        try:
            return models_mod.resolve_weights(model, ratio, level)
        except models_mod.InvalidModelError as exc:
            raise EngineError(str(exc)) from exc
        except FileNotFoundError as exc:
            if model == models_mod.DEFAULT_MODEL:
                fallback = self._auto_fallback(ratio, level)
                if fallback is not None:
                    logger.warning(
                        "auto 首选模型权重缺失，降级为 %s", fallback["model_id"]
                    )
                    return fallback
            raise EngineError(str(exc)) from exc

    def _auto_fallback(self, ratio: float, level: int) -> Optional[dict]:
        available = set(models_mod.available_model_ids())
        for candidate in _AUTO_FALLBACK_ORDER:
            if candidate not in available:
                continue
            try:
                return models_mod.resolve_weights(candidate, ratio, level)
            except (FileNotFoundError, models_mod.InvalidModelError):
                continue
        return None

    def _get_module(self, resolved: dict):
        """Load (and cache) the concrete network for a resolved checkpoint."""
        key = f"{resolved['family']}:{resolved['weight_file']}"
        module = self._modules.get(key)
        if module is not None:
            return module
        weight_path = Path(resolved["weight_paths"][0])
        scale = int(resolved["native_scale"])
        try:
            if resolved["family"] == "rrdb":
                from .rrdbnet import load_rrdbnet  # lazy: keeps PIL backend torch-free

                module = load_rrdbnet(
                    weight_path,
                    scale,
                    self._device,
                    num_block=int(resolved.get("num_block", 23)),
                )
            else:
                from .cugan import load_cugan  # lazy

                module = load_cugan(scale, weight_path, self._device)
        except Exception as exc:  # noqa: BLE001 - surfaced as EngineError
            raise EngineError(f"模型加载失败（{resolved['weight_file']}）：{exc}") from exc
        self._modules[key] = module
        logger.info("模型已加载：%s（%s）", resolved["weight_file"], self._device)
        return module

    # -- public API ---------------------------------------------------------
    def resolve_model(self, model: str, ratio: float, denoise_level: int = 0) -> dict:
        _entry, level = _validate_model_and_level(model, False, denoise_level)
        resolved = self._resolve_weights(model, ratio, level)
        return {
            "id": model,
            "label": resolved["label"],
            "used": _display_name(resolved),
            "family": resolved["family"],
            "native_scale": int(resolved["native_scale"]),
            "weight_file": resolved["weight_file"],
            "denoise_level": int(resolved.get("denoise_level") or 0),
        }

    def upscale(
        self,
        img: Image.Image,
        out_w: int,
        out_h: int,
        *,
        model: str = "auto",
        denoise: bool = False,
        denoise_level: int = 0,
        tile: int = 0,
        progress_cb: Optional[ProgressCB] = None,
    ) -> Image.Image:
        if not isinstance(img, Image.Image):  # pragma: no cover - defensive
            raise TypeError("img 必须是 PIL.Image.Image")
        out_w, out_h = _validate_out_size(out_w, out_h)
        cb = _ProgressGuard(progress_cb)

        src_rgb, alpha = _split_alpha(img)
        src_w, src_h = src_rgb.size
        same = (src_w, src_h) == (out_w, out_h)
        out_area = out_w * out_h

        try:
            if same or out_area < src_w * src_h:
                # Downscale / same size: never run a network (CONTRACT §3.4).
                cb(0.02)
                if same:
                    rgb = src_rgb.copy()
                    cb(0.6)
                else:
                    rgb = _resize_rgb(src_rgb, out_w, out_h)
                    cb(0.85)
                result = _merge_alpha(rgb, alpha, out_w, out_h)
                cb(1.0)
                return result

            # ---- upscale: run the network, then LANCZOS to the exact target ----
            ratio = max(out_w / float(src_w), out_h / float(src_h))
            _entry, level = _validate_model_and_level(model, denoise, denoise_level)
            resolved = self._resolve_weights(model, ratio, level)
            tile_size = _normalise_tile(tile)
            sr_rgb: Optional[Image.Image] = None
            try:
                with self._lock:
                    module = self._get_module(resolved)
                    sr_rgb = self._run_sr(
                        src_rgb, resolved, module, tile_size, cb, denoise=denoise
                    )
            except Exception as exc:  # noqa: BLE001 - cancellation vs failure below
                if cb.fired:
                    raise  # progress_cb raised (usually RuntimeError("cancelled"))
                logger.warning(
                    "模型推理失败（%s），本张图片降级为 PIL 缩放：%s",
                    resolved.get("weight_file"), exc,
                )
                sr_rgb = None

            if sr_rgb is None:
                rgb = _resize_rgb(src_rgb, out_w, out_h, unsharp=PIL_UNSHARP)
            else:
                rgb = _resize_rgb(sr_rgb, out_w, out_h, unsharp=SR_UNSHARP)
            cb(0.98)
            result = _merge_alpha(rgb, alpha, out_w, out_h)
            cb(1.0)
            return result
        finally:
            self._empty_cache()

    def close(self) -> None:
        with self._lock:
            self._modules.clear()
        self._empty_cache()

    # -- internals ----------------------------------------------------------
    def _empty_cache(self) -> None:
        if self._device.type != "cuda":  # pragma: no cover - CPU path
            return
        try:
            self._torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - defensive
            pass

    def _opencv_denoise(self, img: Image.Image) -> Image.Image:
        """RRDB pre-denoise: OpenCV non-local means (API.md §14.3)."""
        cv2 = _import_cv2()
        if cv2 is None:
            return img
        try:
            import numpy as np  # noqa: PLC0415 - lazy

            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
            return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
        except Exception:  # pragma: no cover - cv2 is optional
            logger.warning("OpenCV 降噪失败，已忽略", exc_info=True)
            return img

    def _run_sr(
        self,
        src_rgb: Image.Image,
        resolved: dict,
        module,
        tile: int,
        progress: _ProgressGuard,
        *,
        denoise: bool,
    ) -> Image.Image:
        """Run the tiled network and return the SR image (``scale * src``)."""
        import numpy as np  # noqa: PLC0415 - everything below needs numpy

        source = src_rgb
        if resolved["family"] == "rrdb" and denoise and self.denoise_available:
            source = self._opencv_denoise(source)
        progress(0.02)

        src_np = np.asarray(source, dtype=np.uint8)
        height, width = src_np.shape[:2]
        scale = int(resolved["native_scale"])
        out_np = np.empty((height * scale, width * scale, 3), dtype=np.uint8)

        if resolved["family"] == "rrdb":
            def run(current_tile: int) -> None:
                self._tiled_rrdb(
                    src_np, module, scale, current_tile, out_np, progress, 0.05, 0.92
                )
        else:
            def run(current_tile: int) -> None:
                self._tiled_cugan(
                    src_np, module, scale, current_tile, out_np, progress, 0.05, 0.92
                )

        _run_with_oom_retry(run, tile, min_tile=MIN_TILE, empty_cache=self._empty_cache)
        return Image.fromarray(out_np, "RGB")

    @staticmethod
    def _tile_grid(width: int, height: int, tile: int) -> tuple[int, int, int]:
        tiles_x = max(1, math.ceil(width / tile))
        tiles_y = max(1, math.ceil(height / tile))
        return tiles_x, tiles_y, tiles_x * tiles_y

    def _tiled_rrdb(
        self,
        src_np,
        module,
        scale: int,
        tile: int,
        out_np,
        progress: _ProgressGuard,
        progress_lo: float,
        progress_hi: float,
    ) -> None:
        """Tiled RRDB forward pass; writes the SR result into ``out_np`` (H, W, 3 uint8)."""
        import numpy as np  # noqa: PLC0415 - lazy

        torch = self._torch
        height, width = src_np.shape[:2]
        pad = TILE_OVERLAP
        align = _UNSHUFFLE_ALIGN.get(int(scale), 1)
        device = self._device
        tiles_x, tiles_y, total = self._tile_grid(width, height, tile)
        done = 0

        with torch.inference_mode():
            for tile_y in range(tiles_y):
                in_y0 = tile_y * tile
                in_y1 = min(in_y0 + tile, height)
                for tile_x in range(tiles_x):
                    in_x0 = tile_x * tile
                    in_x1 = min(in_x0 + tile, width)

                    # Context window for this tile (clamped to the image border).
                    pad_y0 = max(in_y0 - pad, 0)
                    pad_y1 = min(in_y1 + pad, height)
                    pad_x0 = max(in_x0 - pad, 0)
                    pad_x1 = min(in_x1 + pad, width)

                    # np.array() copies, giving torch a writable input buffer.
                    patch = np.array(
                        src_np[pad_y0:pad_y1, pad_x0:pad_x1, :], dtype=np.uint8
                    )
                    if align > 1:
                        extra_y = (-patch.shape[0]) % align
                        extra_x = (-patch.shape[1]) % align
                        if extra_y or extra_x:
                            patch = np.pad(
                                patch, ((0, extra_y), (0, extra_x), (0, 0)), mode="edge"
                            )

                    # Contiguous CPU tensor: already normalized to [0,1] CHW.
                    # (RRDBNet's pixel_unshuffle uses .view() and needs contiguity.)
                    tensor = (
                        torch.from_numpy(patch)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(dtype=torch.float32)
                        .contiguous()
                        .div_(255.0)
                        .to(device=device, non_blocking=True)
                    )
                    tile_out = module(tensor)

                    # Crop the valid (non-overlap) part out of this tile's output.
                    # ``local_*`` are coordinates inside tile_out; ``out_*`` are
                    # the destination coordinates in the full-resolution mosaic.
                    local_y0 = (in_y0 - pad_y0) * scale
                    local_x0 = (in_x0 - pad_x0) * scale
                    local_y1 = local_y0 + (in_y1 - in_y0) * scale
                    local_x1 = local_x0 + (in_x1 - in_x0) * scale
                    out_y0, out_y1 = in_y0 * scale, in_y1 * scale
                    out_x0, out_x1 = in_x0 * scale, in_x1 * scale
                    chunk = tile_out[0, :, local_y0:local_y1, local_x0:local_x1]
                    chunk = (
                        chunk.clamp_(0.0, 1.0)
                        .mul_(255.0)
                        .round_()
                        .to(torch.uint8)
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
                    out_np[out_y0:out_y1, out_x0:out_x1, :] = chunk

                    done += 1
                    progress(progress_lo + (progress_hi - progress_lo) * done / total)

    def _tiled_cugan(
        self,
        src_np,
        module,
        scale: int,
        tile: int,
        out_np,
        progress: _ProgressGuard,
        progress_lo: float,
        progress_hi: float,
    ) -> None:
        """Tiled Real-CUGAN forward pass (uint8 network output -> uint8 mosaic)."""
        import numpy as np  # noqa: PLC0415 - lazy

        torch = self._torch
        height, width = src_np.shape[:2]
        pad = CUGAN_TILE_OVERLAP  # CONTRACT §3.4: CUGAN pad >= 32
        device = self._device
        tiles_x, tiles_y, total = self._tile_grid(width, height, tile)
        done = 0

        with torch.inference_mode():
            for tile_y in range(tiles_y):
                in_y0 = tile_y * tile
                in_y1 = min(in_y0 + tile, height)
                for tile_x in range(tiles_x):
                    in_x0 = tile_x * tile
                    in_x1 = min(in_x0 + tile, width)

                    pad_y0 = max(in_y0 - pad, 0)
                    pad_y1 = min(in_y1 + pad, height)
                    pad_x0 = max(in_x0 - pad, 0)
                    pad_x1 = min(in_x1 + pad, width)

                    patch = np.array(
                        src_np[pad_y0:pad_y1, pad_x0:pad_x1, :], dtype=np.uint8
                    )
                    # CUGAN pads its input by 14/18/19 px reflect; very small
                    # edge patches must be grown to stay valid.  Padding only to
                    # the bottom/right keeps the tile origin (crop coordinates)
                    # unchanged.
                    if (
                        patch.shape[0] < CUGAN_MIN_INPUT
                        or patch.shape[1] < CUGAN_MIN_INPUT
                    ):
                        patch = np.pad(
                            patch,
                            (
                                (0, max(0, CUGAN_MIN_INPUT - patch.shape[0])),
                                (0, max(0, CUGAN_MIN_INPUT - patch.shape[1])),
                                (0, 0),
                            ),
                            mode="edge",
                        )
                    tensor = (
                        torch.from_numpy(patch)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(dtype=torch.float32)
                        .contiguous()
                        .div_(255.0)
                        .to(device=device, non_blocking=True)
                    )
                    # Real-CUGAN v3 signature; output is uint8 NCHW [0,255].
                    tile_out = module(tensor, 0, 0, 1.0, False)

                    local_y0 = (in_y0 - pad_y0) * scale
                    local_x0 = (in_x0 - pad_x0) * scale
                    local_y1 = local_y0 + (in_y1 - in_y0) * scale
                    local_x1 = local_x0 + (in_x1 - in_x0) * scale
                    out_y0, out_y1 = in_y0 * scale, in_y1 * scale
                    out_x0, out_x1 = in_x0 * scale, in_x1 * scale
                    chunk = tile_out[0, :, local_y0:local_y1, local_x0:local_x1]
                    if chunk.dtype != torch.uint8:  # defensive: never rescale uint8
                        chunk = chunk.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
                    chunk = chunk.permute(1, 2, 0).cpu().numpy()
                    out_np[out_y0:out_y1, out_x0:out_x1, :] = chunk

                    done += 1
                    progress(progress_lo + (progress_hi - progress_lo) * done / total)


# ---------------------------------------------------------------------------
# Engine factory / singleton cache
# ---------------------------------------------------------------------------
_ENGINE_LOCK = threading.RLock()
_ENGINE_CACHE: dict[str, BaseEngine] = {}


def _build_realesrgan(*, require_cuda: bool = False) -> BaseEngine:
    """Create a multi-model Real-ESRGAN engine or raise :class:`EngineError`."""
    # Cheap checks first: missing weights should not pay for importing torch.
    usable = [
        model_id
        for model_id in models_mod.available_model_ids()
        if model_id != models_mod.DEFAULT_MODEL
    ]
    if not usable:
        raise EngineError(
            "未找到任何超分模型权重（models/ 为空），可运行 scripts/fetch_models.py 下载"
        )
    torch = _import_torch()
    if torch is None:
        raise EngineError("PyTorch 不可用，无法使用 Real-ESRGAN 引擎")
    if require_cuda and not _cuda_available(torch):
        raise EngineError(
            "CUDA 不可用；自动模式将使用 PIL 兼容引擎（如需 CPU 推理请设置 "
            "PIC_POLISH_ENGINE=realesrgan）"
        )
    try:
        return RealESRGANEngine(torch_module=torch)
    except EngineError:
        raise
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise EngineError(f"Real-ESRGAN 引擎初始化失败：{exc}") from exc


def _create_engine(mode: str) -> BaseEngine:
    if mode == "pil":
        return PILEngine()
    if mode == "realesrgan":
        return _build_realesrgan(require_cuda=False)
    # auto: Real-ESRGAN when torch + CUDA + weights are all available, else PIL.
    try:
        return _build_realesrgan(require_cuda=True)
    except EngineError as exc:
        logger.warning("Real-ESRGAN 引擎不可用，自动降级为 PIL 兼容模式：%s", exc)
        return PILEngine()


def get_engine(force: Optional[str] = None) -> BaseEngine:
    """Return the (cached) engine singleton.

    Args:
        force: ``"auto"`` (default, from ``config.ENGINE``), ``"realesrgan"`` or
            ``"pil"``.  Forcing ``realesrgan`` raises :class:`EngineError` when
            it cannot be created; ``auto`` never raises and falls back to PIL.
    """
    mode = (force if force is not None else config.ENGINE) or "auto"
    mode = str(mode).strip().lower()
    if mode not in _ENGINE_MODES:
        raise EngineError(f"未知的引擎模式：{force!r}（可选 auto/realesrgan/pil）")
    with _ENGINE_LOCK:
        engine = _ENGINE_CACHE.get(mode)
        if engine is not None:
            return engine
        engine = _create_engine(mode)
        _ENGINE_CACHE[mode] = engine
        if mode == "auto" and engine.info.backend == "realesrgan":
            # Let a later forced "realesrgan" reuse the already-loaded engine.
            _ENGINE_CACHE.setdefault("realesrgan", engine)
        return engine


def engine_info() -> EngineInfo:
    """Return the :class:`EngineInfo` of the active engine (for /api/health)."""
    return get_engine().info


def list_models() -> dict:
    """Forward :func:`app.models.list_models` (API.md §14.1); loads no model."""
    return models_mod.list_models()


def shutdown_engines() -> None:
    """Close every cached engine and clear the cache (idempotent)."""
    closed_realesrgan = False
    with _ENGINE_LOCK:
        seen: set[int] = set()
        for engine in _ENGINE_CACHE.values():
            if id(engine) in seen:
                continue
            seen.add(id(engine))
            if getattr(getattr(engine, "info", None), "backend", None) == "realesrgan":
                closed_realesrgan = True
            try:
                engine.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("关闭引擎失败", exc_info=True)
        _ENGINE_CACHE.clear()
    # Avoid importing torch when only the PIL backend was ever used.
    if closed_realesrgan:
        _empty_cuda_cache()
