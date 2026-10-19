"""Central configuration for PicPolish.

All paths are absolute and derived from the project root, so the service can be
started from any working directory (the ``.bat``/``.sh`` launchers rely on this).

Only the Lead edits this file; every other module treats it as read-only.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser().resolve()
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_NAME = "PicPolish"
VERSION = "1.1.0"

ROOT = _env_path("PIC_POLISH_ROOT", PROJECT_ROOT)
MODELS_DIR = _env_path("PIC_POLISH_MODELS_DIR", ROOT / "models")
CACHE_DIR = _env_path("PIC_POLISH_CACHE_DIR", ROOT / "cache")
UPLOAD_DIR = CACHE_DIR / "uploads"
RESULT_DIR = CACHE_DIR / "results"
TMP_DIR = CACHE_DIR / "tmp"
WEB_DIR = _env_path("PIC_POLISH_WEB_DIR", ROOT / "web")
LOG_DIR = _env_path("PIC_POLISH_LOG_DIR", ROOT / "logs")
RUNTIME_DIR = _env_path("PIC_POLISH_RUNTIME_DIR", ROOT / ".runtime")

HOST = os.environ.get("PIC_POLISH_HOST", "127.0.0.1")
PORT = _env_int("PIC_POLISH_PORT", 8765)
# auto | realesrgan | pil
ENGINE = os.environ.get("PIC_POLISH_ENGINE", "auto").strip().lower()
TILE = _env_int("PIC_POLISH_TILE", 256)

MAX_FILES = _env_int("PIC_POLISH_MAX_FILES", 20)
MAX_FILE_MB = _env_int("PIC_POLISH_MAX_FILE_MB", 50)
MAX_TOTAL_MB = _env_int("PIC_POLISH_MAX_TOTAL_MB", 200)
MAX_PIXELS = _env_int("PIC_POLISH_MAX_PIXELS", 20_000_000)
MIN_SIDE = 16
MAX_SIDE = 16384

DEFAULT_ALIGN = 8
MIN_ALIGN = 2
MAX_ALIGN = 64
JPEG_QUALITY = _env_int("PIC_POLISH_JPEG_QUALITY", 95)
KEEP_HOURS = float(os.environ.get("PIC_POLISH_KEEP_HOURS", "24"))
CLEAN_ON_EXIT = os.environ.get("PIC_POLISH_CLEAN_ON_EXIT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MODEL_X4 = "RealESRGAN_x4plus.pth"
MODEL_X2 = "RealESRGAN_x2plus.pth"


def ensure_dirs() -> None:
    """Create every runtime directory (idempotent)."""
    for path in (MODELS_DIR, CACHE_DIR, UPLOAD_DIR, RESULT_DIR, TMP_DIR, LOG_DIR, RUNTIME_DIR):
        path.mkdir(parents=True, exist_ok=True)


def cache_image_bytes() -> int:
    """Total bytes currently stored under the cache image folders."""
    total = 0
    for folder in (UPLOAD_DIR, RESULT_DIR, TMP_DIR):
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    return total


def clear_cache_images() -> dict:
    """Delete everything inside uploads/results/tmp but keep the directories.

    Returns ``{"removed_files": int, "freed_bytes": int}``.
    Used by the server shutdown hook; the stop scripts implement their own cleanup.
    """
    removed = 0
    freed = 0
    for folder in (UPLOAD_DIR, RESULT_DIR, TMP_DIR):
        if not folder.exists():
            continue
        for child in sorted(folder.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            try:
                if child.is_file() or child.is_symlink():
                    freed += child.stat().st_size if child.exists() else 0
                    child.unlink(missing_ok=True)
                    removed += 1
                elif child.is_dir():
                    child.rmdir()
            except OSError:
                continue
    return {"removed_files": removed, "freed_bytes": freed}
