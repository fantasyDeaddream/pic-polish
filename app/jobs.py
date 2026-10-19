"""Job queue, background worker and result storage for PicPolish.

Contract: ``docs/CONTRACT.md`` section 3.5 and ``docs/API.md`` sections 6-10.

Design
------
* One worker thread processes jobs FIFO so the GPU is used serially.
* Everything a request needs is kept in memory (a restart drops the queue, the
  startup/temporary cleanup removes stale folders).
* Every mutation of shared state happens under a single ``threading.RLock``; the
  public ``get()``/``list_jobs()`` methods return detached plain dicts so route
  handlers never race with the worker.
* The engine module (``app/engine.py``) is imported lazily so this module can be
  imported (and unit tested with a fake engine) on machines without torch.
"""
from __future__ import annotations

import io
import os
import queue
import re
import shutil
import sys
import threading
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from PIL import Image, ImageOps, UnidentifiedImageError

from . import config
from .target import InvalidTarget, plan_many, plan_target

# ---------------------------------------------------------------------------
# Errors (``status_code`` is mapped to HTTP by ``app.main``)
# ---------------------------------------------------------------------------


class JobError(Exception):
    """Base error raised by the job layer; carries the HTTP status to use."""

    status_code = 400


class InvalidRequestError(JobError):
    """Malformed request (bad tier/align/options, undecodable image, ...)."""

    status_code = 400


class UnsupportedMediaError(JobError):
    """Uploaded file is not a supported image type."""

    status_code = 415


class FileTooLargeError(JobError):
    """A single upload or the whole batch exceeds the configured size limit."""

    status_code = 413


class JobNotFoundError(JobError):
    """Unknown job id."""

    status_code = 404


class JobConflictError(JobError):
    """The job is in a state that does not allow the requested operation."""

    status_code = 409


class InternalError(JobError):
    """Unexpected server side failure while creating/processing a job."""

    status_code = 500


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

IMAGE_CONTENT_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}
OUTPUT_FORMATS = ("auto", "png", "jpg", "webp")
_FORMAT_ALIASES = {"jpeg": "jpg"}

_TERMINAL_STATUSES = ("done", "error", "cancelled")
_ACTIVE_STATUSES = ("queued", "running")

_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

_UNSAFE_CHARS = re.compile(r"[^\w\-.]", re.UNICODE)
_MULTI_UNDERSCORE = re.compile(r"_{2,}")


def _now() -> datetime:
    """Local timezone aware ``datetime`` (API timestamps look like +08:00)."""
    return datetime.now().astimezone()


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat(timespec="seconds")


def sanitize_filename(name: Any, *, default: str = "image", max_length: int = 120) -> str:
    """Return a path safe file name (no separators, no traversal, sane length)."""
    text = str(name or "").strip()
    # Drop any directory part, keep the basename only (handles / and \).
    text = re.split(r"[\\/]+", text)[-1] if text else ""
    text = text.replace("\x00", "")
    text = _UNSAFE_CHARS.sub("_", text)
    text = _MULTI_UNDERSCORE.sub("_", text).strip().strip(" .")
    if not text or text in (".", ".."):
        text = default
    if len(text) > max_length:
        stem, ext = os.path.splitext(text)
        ext = ext[:16]
        keep = max(1, max_length - len(ext))
        stem = stem[:keep].rstrip(" .")
        text = f"{stem}{ext}"
        if not text or text == ext:
            text = f"{default}{ext}"
    stem, _ext = os.path.splitext(text)
    if stem.upper() in _WINDOWS_RESERVED:
        text = f"_{text}"
    return text or default


def content_disposition(filename: str, inline: bool = False) -> str:
    """Build a RFC 6266/5987 safe Content-Disposition header value.

    ``filename`` may be non-ASCII (Chinese file names are supported by writing
    both an ASCII fallback and the RFC 5987 ``filename*`` parameter).  When the
    ASCII fallback would lose the whole stem (e.g. ``我的照片.jpg`` -> ``.jpg``)
    a readable default stem is used so the header stays valid:
    ``filename="download.jpg"``.
    """
    disposition = "inline" if inline else "attachment"
    # Build the ASCII fallback from the *stem* only: stripping non-ASCII from the
    # whole name would turn "我的照片.jpg" into ".jpg" (a hidden file with no
    # extension), which is not a usable download name.
    suffix = "".join(ch for ch in Path(filename).suffix if 32 <= ord(ch) < 127)
    ascii_stem = Path(filename).stem.encode("ascii", "ignore").decode("ascii")
    ascii_stem = "".join(ch for ch in ascii_stem if 32 <= ord(ch) < 127)
    ascii_stem = ascii_stem.replace('"', "").replace("\\", "").strip(" .") or "download"
    fallback = f"{ascii_stem}{suffix}"
    try:
        filename.encode("ascii")
        if fallback == filename:
            return f'{disposition}; filename="{filename}"'
    except UnicodeEncodeError:
        pass
    quoted = quote(filename, safe="")
    return f'{disposition}; filename="{fallback}"; filename*=UTF-8\'\'{quoted}'


def sanitize_download_filename(requested: str | None, extension: str) -> str | None:
    """Sanitize the user supplied ``?filename=`` while forcing the real extension."""
    if not requested:
        return None
    clean = sanitize_filename(requested, default="image")
    stem, _old = os.path.splitext(clean)
    stem = sanitize_filename(stem or "image", default="image")
    return f"{stem}.{extension}"


def _unique_name(used: set[str], candidate: str) -> str:
    """Return ``candidate`` or ``stem_2.ext``/``stem_3.ext`` if already used."""
    if candidate.casefold() not in used:
        return candidate
    stem, ext = os.path.splitext(candidate)
    counter = 2
    while True:
        trial = f"{stem}_{counter}{ext}"
        if trial.casefold() not in used:
            return trial
        counter += 1


def _dir_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _info_to_dict(info: Any) -> dict | None:
    """Normalize an ``EngineInfo`` dataclass / dict into a JSON ready dict."""
    if info is None:
        return None
    raw: dict | None = None
    if isinstance(info, dict):
        raw = info
    else:
        to_dict = getattr(info, "to_dict", None)
        if callable(to_dict):
            try:
                candidate = to_dict()
            except Exception:  # noqa: BLE001 - fall back to attribute probing
                candidate = None
            if isinstance(candidate, dict):
                raw = candidate
        if raw is None:
            raw = {
                key: getattr(info, key, None)
                for key in ("backend", "device", "gpu", "model", "native_scale")
            }
    try:
        native_scale = int(raw.get("native_scale") or 1)
    except (TypeError, ValueError):
        native_scale = 1
    return {
        "backend": raw.get("backend") or "pil",
        "device": raw.get("device") or "cpu",
        "gpu": raw.get("gpu"),
        "model": raw.get("model"),
        "native_scale": native_scale,
    }


_ENGINE_INFO_LOCK = threading.Lock()
_ENGINE_INFO_CACHE: dict | None = None

_PIL_FALLBACK_INFO = {
    "backend": "pil",
    "device": "cpu",
    "gpu": None,
    "model": None,
    "native_scale": 1,
}


def engine_info_dict(refresh: bool = False) -> dict:
    """Return the current engine info as a plain dict (never raises, cached)."""
    global _ENGINE_INFO_CACHE
    if not refresh:
        with _ENGINE_INFO_LOCK:
            cached = _ENGINE_INFO_CACHE
        if cached is not None:
            return dict(cached)
    info: dict | None = None
    try:
        from .engine import engine_info as _engine_info  # lazy import

        info = _info_to_dict(_engine_info())
    except Exception:  # noqa: BLE001 - engine missing/broken must not break health
        info = None
    if info is None:
        info = dict(_PIL_FALLBACK_INFO)
    with _ENGINE_INFO_LOCK:
        _ENGINE_INFO_CACHE = dict(info)
    return dict(info)


def get_engine(force: str | None = None):
    """Lazily create/return the (singleton) upscale engine.

    Kept as a module level function so unit tests can monkeypatch it with a fake
    engine and never touch torch/the GPU.
    """
    from .engine import get_engine as _get_engine  # lazy import

    return _get_engine(force)


# ---------------------------------------------------------------------------
# Model registry helpers (``app/models.py``, see docs/CONTRACT.md §3.7)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "auto"
_FALLBACK_MODEL_IDS = (
    "auto",
    "realesrgan-x4plus",
    "realesrgan-x2plus",
    "realesrgan-anime6b",
    "realcugan",
    "realcugan-conservative",
)
_FALLBACK_MODEL_LABELS = {
    "auto": "智能推荐",
    "realesrgan-x4plus": "Real-ESRGAN 通用照片",
    "realesrgan-x2plus": "Real-ESRGAN 低倍率",
    "realesrgan-anime6b": "Real-ESRGAN 动漫",
    "realcugan": "Real-CUGAN（动漫/视频）",
    "realcugan-conservative": "Real-CUGAN 保守",
}


def list_models() -> dict:
    """Return the model registry payload (API.md §14.1) without loading models.

    ``app.models.list_models()`` is the primary source; the engine forwards it,
    so fall back to the engine / a minimal stub when the registry is missing.
    """
    try:
        from . import models as models_module  # lazy import, light module

        payload = models_module.list_models()
        if isinstance(payload, dict) and isinstance(payload.get("models"), list):
            return payload
    except Exception:  # noqa: BLE001 - fall through to the engine/stub
        pass
    try:
        from .engine import list_models as engine_list_models  # lazy import

        payload = engine_list_models()
        if isinstance(payload, dict) and isinstance(payload.get("models"), list):
            return payload
    except Exception:  # noqa: BLE001 - engine may not expose it yet
        pass
    return {"default": DEFAULT_MODEL, "models": []}


def models_status() -> dict:
    """Return ``{"default": "auto", "available": [...]}`` for ``/api/health``."""
    payload = list_models()
    available: list[str] = []
    for item in payload.get("models") or []:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if model_id and item.get("available"):
            available.append(str(model_id))
    default = payload.get("default")
    return {"default": str(default) if default else DEFAULT_MODEL, "available": available}


def _model_catalog() -> tuple[dict[str, dict], list[str]]:
    payload = list_models()
    definitions: dict[str, dict] = {}
    for item in payload.get("models") or []:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if model_id:
            definitions[str(model_id)] = item
    return definitions, list(definitions)


def _fallback_model_used(model_id: str, ratio: float) -> str:
    """Best effort ``model_used`` label when the engine has no resolver."""
    if model_id != DEFAULT_MODEL:
        return model_id
    try:
        numeric = float(ratio)
    except (TypeError, ValueError):
        numeric = 1.0
    return "RealESRGAN_x2plus" if numeric <= 2.0 else "RealESRGAN_x4plus"


class CancelToken:
    """Thread safe cooperative cancellation flag handed to jobs."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        """Raise :class:`_Cancelled` if cancellation was requested."""
        if self._event.is_set():
            raise _Cancelled()


class _Cancelled(RuntimeError):
    def __init__(self, message: str = "cancelled") -> None:
        super().__init__(message)


# ---------------------------------------------------------------------------
# Internal job data
# ---------------------------------------------------------------------------


class _JobFile:
    __slots__ = (
        "index", "name", "size", "original", "plan", "upload_path", "weight",
        "output_choice", "status", "progress", "output_size", "output_format",
        "filename", "result_path", "note", "error",
        "model", "model_used", "denoise_level",
    )

    def __init__(
        self,
        *,
        index: int,
        name: str,
        size: int,
        original: dict,
        plan: dict,
        upload_path: Path,
        output_choice: str,
        model: str = DEFAULT_MODEL,
        denoise_level: int = 0,
    ) -> None:
        self.index = index
        self.name = name
        self.size = size
        self.original = original
        self.plan = plan
        self.upload_path = upload_path
        output = plan.get("output") or {}
        try:
            self.weight = max(1, int(output.get("width") or 1) * int(output.get("height") or 1))
        except (TypeError, ValueError):
            self.weight = 1
        self.output_choice = output_choice
        self.status = "pending"
        self.progress = 0.0
        self.output_size: int | None = None
        self.output_format: str | None = None
        self.filename: str | None = None
        self.result_path: Path | None = None
        self.note: str | None = None
        self.error: str | None = None
        self.model = model
        self.model_used: str | None = None
        self.denoise_level = int(denoise_level or 0)


class _Job:
    __slots__ = (
        "job_id", "target", "align", "options", "created_at", "files", "engine",
        "token", "status", "started_at", "finished_at", "error", "current_index",
        "processing", "result_names", "model", "requested_denoise_level",
    )

    def __init__(
        self,
        *,
        job_id: str,
        target: dict,
        align: int,
        options: dict,
        files: list[_JobFile],
        model: dict | None = None,
        requested_denoise_level: int = 0,
    ) -> None:
        self.job_id = job_id
        self.target = target
        self.align = align
        self.options = options
        self.created_at = _now()
        self.files = files
        self.engine: dict = {}
        self.token = CancelToken()
        self.status = "queued"
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.error: str | None = None
        self.current_index: int | None = None
        self.processing = False
        self.result_names: set[str] = set()
        self.model = dict(model) if model else {
            "id": DEFAULT_MODEL,
            "label": _FALLBACK_MODEL_LABELS[DEFAULT_MODEL],
            "denoise": bool(options.get("denoise")),
            "denoise_level": int(options.get("denoise_level") or 0),
        }
        self.requested_denoise_level = int(requested_denoise_level or 0)


# ---------------------------------------------------------------------------
# Job manager
# ---------------------------------------------------------------------------


class JobManager:
    """FIFO job queue with a single processing worker (GPU serial)."""

    def __init__(
        self,
        *,
        keep_hours: float | None = None,
        cleanup_interval: float = 3600.0,
    ) -> None:
        try:
            config.ensure_dirs()
        except OSError:  # pragma: no cover - defensive, paths should be writable
            pass
        self._lock = threading.RLock()
        self._jobs: dict[str, _Job] = {}
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._keep_hours = float(config.KEEP_HOURS if keep_hours is None else keep_hours)
        self._cleanup_interval = max(1.0, float(cleanup_interval))
        self._stop = threading.Event()
        self._closed = False
        self._ttl_thread = threading.Thread(
            target=self._ttl_loop, name="picpolish-ttl", daemon=True
        )
        self._worker = threading.Thread(
            target=self._worker_loop, name="picpolish-worker", daemon=True
        )
        self._worker.start()
        self._ttl_thread.start()

    # -- public API --------------------------------------------------------

    def submit(
        self,
        files: "list[tuple[str, bytes]]",
        target: str,
        options: dict | None = None,
    ) -> dict:
        """Validate, persist uploads and enqueue a new job.

        Returns the API job dict (``docs/API.md`` section 6) plus
        ``queue_position``.
        """
        options = dict(options or {})
        if not files:
            raise InvalidRequestError("请至少上传 1 张图片")
        if len(files) > config.MAX_FILES:
            raise InvalidRequestError(f"一次最多 {config.MAX_FILES} 张图片")
        if not isinstance(target, str) or not target.strip():
            raise InvalidRequestError("缺少 target 参数（可选 720p/1080p/2k/4k）")
        target = target.strip()

        tier = self._validate_target(target)
        align = self._validate_align(options.get("align", config.DEFAULT_ALIGN))
        out_format = self._validate_format(options.get("format", "auto"))
        quality = self._validate_quality(options.get("quality", config.JPEG_QUALITY))
        denoise = self._validate_bool(options.get("denoise", False), "denoise")
        model_id, model_definition = self._validate_model(
            options.get("model", DEFAULT_MODEL)
        )
        requested_denoise_level = self._validate_denoise_level(
            options.get("denoise_level", 0)
        )
        denoise_level = self._effective_denoise_level(
            model_id, model_definition, denoise, requested_denoise_level
        )
        model_label = str(
            model_definition.get("label")
            or _FALLBACK_MODEL_LABELS.get(model_id)
            or model_id
        )
        tile = self._validate_tile(options.get("tile", 0))

        max_file_bytes = max(1, config.MAX_FILE_MB) * 1024 * 1024
        max_total_bytes = max(1, config.MAX_TOTAL_MB) * 1024 * 1024

        prepared: list[dict] = []
        total_bytes = 0
        for index, item in enumerate(files):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise InvalidRequestError(f"第 {index + 1} 个上传文件格式错误")
            raw_name, data = item
            raw_name = str(raw_name or f"image_{index + 1}")
            if not isinstance(data, (bytes, bytearray)):
                raise InvalidRequestError(f"第 {index + 1} 个上传文件内容无效")
            data = bytes(data)
            display_name = sanitize_filename(raw_name, default=f"image_{index + 1}")
            suffix = Path(display_name).suffix.lower()
            if suffix not in config.ALLOWED_SUFFIXES:
                raise UnsupportedMediaError(f"不支持的文件类型：{raw_name}")
            if not data:
                raise InvalidRequestError(f"空文件：{display_name}")
            if len(data) > max_file_bytes:
                raise FileTooLargeError(
                    f"单个文件超过 {config.MAX_FILE_MB}MB 限制：{display_name}"
                )
            total_bytes += len(data)
            probe = self._probe_image(display_name, data)
            prepared.append(
                {
                    "index": index,
                    "name": display_name,
                    "size": len(data),
                    "data": data,
                    "probe": probe,
                }
            )
        if total_bytes > max_total_bytes:
            raise FileTooLargeError(f"文件总大小超过 {config.MAX_TOTAL_MB}MB 限制")

        images = [
            {
                "name": item["name"],
                "width": item["probe"]["width"],
                "height": item["probe"]["height"],
            }
            for item in prepared
        ]
        try:
            plans = plan_many(images, target, align=align)
        except InvalidTarget as exc:  # bad dimensions / output too large
            raise InvalidRequestError(str(exc)) from exc

        engine = engine_info_dict()
        job_id = self._new_job_id()
        upload_dir = config.UPLOAD_DIR / job_id
        result_dir = config.RESULT_DIR / job_id
        tmp_dir = config.TMP_DIR / job_id
        try:
            upload_dir.mkdir(parents=True, exist_ok=True)
            result_dir.mkdir(parents=True, exist_ok=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise InternalError(f"无法创建缓存目录：{exc}") from exc

        job_files: list[_JobFile] = []
        try:
            for item, plan in zip(prepared, plans):
                upload_path = upload_dir / f"{item['index']}_{item['name']}"
                upload_path.write_bytes(item["data"])
                chosen = self._choose_output_format(out_format, item["probe"])
                job_files.append(
                    _JobFile(
                        index=item["index"],
                        name=item["name"],
                        size=item["size"],
                        original=dict(plan["origin"]),
                        plan=plan,
                        upload_path=upload_path,
                        output_choice=chosen,
                        model=model_id,
                        denoise_level=denoise_level,
                    )
                )
        except OSError as exc:
            self._remove_job_dirs(job_id)
            raise InternalError(f"上传文件保存失败：{exc}") from exc
        except Exception:
            self._remove_job_dirs(job_id)
            raise

        job_target = {
            "id": tier["id"],
            "label": tier["label"],
            "rule": tier["rule"],
            "value": tier["value"],
            "align": align,
        }
        job_options = {
            "align": align,
            "model": model_id,
            "denoise": denoise,
            "denoise_level": denoise_level,
            "format": out_format,
            "quality": quality,
            "tile": tile,
        }
        job_model = {
            "id": model_id,
            "label": model_label,
            "denoise": denoise,
            "denoise_level": denoise_level,
        }
        job = _Job(
            job_id=job_id,
            target=job_target,
            align=align,
            options=job_options,
            files=job_files,
            model=job_model,
            requested_denoise_level=requested_denoise_level,
        )
        job.engine = dict(engine)

        with self._lock:
            queue_position = sum(
                1 for other in self._jobs.values() if other.status in _ACTIVE_STATUSES
            )
            self._jobs[job_id] = job
        self._queue.put(job_id)
        snapshot = self._job_public(job)
        snapshot["queue_position"] = queue_position
        return snapshot

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return self._job_public(job)

    def list_jobs(self) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda item: item.created_at)
        return [self._job_summary(job) for job in jobs]

    def stats(self) -> dict:
        with self._lock:
            count = len(self._jobs)
        return {
            "jobs": count,
            "result_bytes": _dir_bytes(config.RESULT_DIR),
            "upload_bytes": _dir_bytes(config.UPLOAD_DIR),
        }

    def result_path(self, job_id: str, index: int) -> Path | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if index < 0 or index >= len(job.files):
                return None
            job_file = job.files[index]
            if job_file.status != "done" or job_file.result_path is None:
                return None
            path = job_file.result_path
        if not path.is_file():
            return None
        return path

    def job_target(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job.target) if job is not None else None

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(f"任务不存在：{job_id}")
            if job.status in _TERMINAL_STATUSES:
                raise JobConflictError(f"任务已结束（{job.status}），无法取消")
            job.token.cancel()
            job.status = "cancelled"
            if job.finished_at is None:
                job.finished_at = _now()
            job.current_index = None
            self._cancel_pending(job)
            return {"job_id": job.job_id, "status": job.status}

    def delete(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(f"任务不存在：{job_id}")
            if job.processing or job.status == "running":
                raise JobConflictError("任务正在处理中，请先取消并等待结束后再删除")
            job.token.cancel()
            self._jobs.pop(job_id, None)
        freed = self._remove_job_dirs(job_id)
        return {"deleted": True, "job_id": job_id, "freed_bytes": freed}

    def build_zip(self, job_id: str) -> tuple[bytes, str]:
        """Build the archive with every finished result; returns (bytes, name)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(f"任务不存在：{job_id}")
            if job.status in _ACTIVE_STATUSES:
                raise JobConflictError("结果尚未生成")
            entries = [
                (item.filename or f"{item.index}_{job.target['id']}", item.result_path)
                for item in job.files
                if item.status == "done" and item.result_path is not None
            ]
            tier = job.target["id"]
        entries = [(name, path) for name, path in entries if path is not None and path.is_file()]
        if not entries:
            raise JobConflictError("结果尚未生成")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            used: set[str] = set()
            for name, path in entries:
                entry = _unique_name(used, sanitize_filename(name, default=f"result.{tier}"))
                used.add(entry.casefold())
                archive.write(path, arcname=entry)
        return buffer.getvalue(), f"PicPolish_{job_id[:8]}.zip"

    def cleanup_expired(self) -> dict:
        """Delete jobs/folders older than ``KEEP_HOURS`` (called by the TTL thread)."""
        keep_seconds = max(0.0, self._keep_hours) * 3600.0
        now = _now()
        expired: list[str] = []
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.processing or job.status in _ACTIVE_STATUSES:
                    continue
                reference = job.finished_at or job.created_at
                try:
                    age = (now - reference).total_seconds()
                except TypeError:  # pragma: no cover - naive datetime fallback
                    age = 0.0
                if age >= keep_seconds:
                    self._jobs.pop(job_id, None)
                    job.token.cancel()
                    expired.append(job_id)
        freed = 0
        for job_id in expired:
            freed += self._remove_job_dirs(job_id)
        freed += self._cleanup_orphan_dirs(keep_seconds)
        return {"removed_jobs": len(expired), "freed_bytes": freed}

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            inflight = [
                job for job in self._jobs.values()
                if job.status in _ACTIVE_STATUSES or job.processing
            ]
        for job in inflight:
            job.token.cancel()
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:  # pragma: no cover - unbounded queue
            pass
        if self._worker.is_alive():
            self._worker.join(timeout=5.0)
        if self._ttl_thread.is_alive():
            self._ttl_thread.join(timeout=2.0)
        _shutdown_loaded_engine()

    # -- validation helpers ------------------------------------------------

    @staticmethod
    def _validate_target(target: str) -> dict:
        from .target import get_tier

        try:
            return get_tier(target)
        except InvalidTarget as exc:
            raise InvalidRequestError(str(exc)) from exc

    @staticmethod
    def _validate_align(value: Any) -> int:
        if isinstance(value, bool):
            raise InvalidRequestError(
                f"align 必须是 {config.MIN_ALIGN}..{config.MAX_ALIGN} 的整数"
            )
        try:
            align = int(value)
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(
                f"align 必须是 {config.MIN_ALIGN}..{config.MAX_ALIGN} 的整数"
            ) from exc
        if align != value and isinstance(value, float):
            raise InvalidRequestError(
                f"align 必须是 {config.MIN_ALIGN}..{config.MAX_ALIGN} 的整数"
            )
        if not (config.MIN_ALIGN <= align <= config.MAX_ALIGN):
            raise InvalidRequestError(
                f"align 必须是 {config.MIN_ALIGN}..{config.MAX_ALIGN} 的整数"
            )
        return align

    @staticmethod
    def _validate_format(value: Any) -> str:
        text = str(value or "auto").strip().lower()
        text = _FORMAT_ALIASES.get(text, text)
        if text not in OUTPUT_FORMATS:
            raise InvalidRequestError(
                f"不支持的输出格式：{value}（可选 auto/png/jpg/webp）"
            )
        return text

    @staticmethod
    def _validate_quality(value: Any) -> int:
        if isinstance(value, bool):
            raise InvalidRequestError("quality 必须是 70..100 的整数")
        try:
            quality = int(value)
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError("quality 必须是 70..100 的整数") from exc
        if not (70 <= quality <= 100):
            raise InvalidRequestError("quality 必须是 70..100 的整数")
        return quality

    @staticmethod
    def _validate_tile(value: Any) -> int:
        if value in (None, ""):
            return 0
        if isinstance(value, bool):
            raise InvalidRequestError("tile 必须是 0 或 64..1024 的整数")
        try:
            tile = int(value)
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError("tile 必须是 0 或 64..1024 的整数") from exc
        if tile != 0 and not (64 <= tile <= 1024):
            raise InvalidRequestError("tile 必须是 0 或 64..1024 的整数")
        return tile

    @staticmethod
    def _validate_bool(value: Any, name: str) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
        raise InvalidRequestError(f"{name} 必须是布尔值（true/false）")

    def _validate_model(self, value: Any) -> tuple[str, dict]:
        """Validate a requested model id against the registry (no model loading)."""
        text = str(value).strip() if value not in (None, "") else DEFAULT_MODEL
        if not text:
            text = DEFAULT_MODEL
        definitions, ids = _model_catalog()
        definition = definitions.get(text)
        if definition is None:
            if text == DEFAULT_MODEL:
                definition = {
                    "id": DEFAULT_MODEL,
                    "label": _FALLBACK_MODEL_LABELS[DEFAULT_MODEL],
                    "family": "rrdb",
                    "available": True,
                }
            else:
                options = "/".join(ids) or "/".join(_FALLBACK_MODEL_IDS)
                raise InvalidRequestError(f"未知的 model：{value}（可选 {options}）")
        if text != DEFAULT_MODEL and definition.get("available") is False:
            raise InvalidRequestError(
                f"模型 {text} 的权重缺失，请运行 scripts/fetch_models.py 下载"
            )
        return text, definition

    @staticmethod
    def _validate_denoise_level(value: Any) -> int:
        if value in (None, ""):
            return 0
        if isinstance(value, bool):
            raise InvalidRequestError("denoise_level 必须是 0..3 的整数")
        try:
            level = int(value)
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError("denoise_level 必须是 0..3 的整数") from exc
        if isinstance(value, float) and value != level:
            raise InvalidRequestError("denoise_level 必须是 0..3 的整数")
        if not (0 <= level <= 3):
            raise InvalidRequestError("denoise_level 必须是 0..3 的整数")
        return level

    @staticmethod
    def _effective_denoise_level(
        model_id: str, definition: dict, denoise: bool, denoise_level: int
    ) -> int:
        """Normalize ``denoise``/``denoise_level`` into the value used by the engine."""
        if model_id == "realcugan-conservative":
            return 0  # conservative weights are fixed, no native level
        family = str(definition.get("family") or "").lower()
        if family == "cugan" or model_id.startswith("realcugan"):
            if denoise_level > 0:
                return denoise_level
            # API.md §14.3: denoise=true without a level == level 2
            return 2 if denoise else 0
        # rrdb family: OpenCV pre-denoise is boolean, native level is 0/1
        return 1 if denoise else 0

    @staticmethod
    def _choose_output_format(requested: str, probe: dict) -> str:
        if requested != "auto":
            return requested
        pil_format = str(probe.get("format") or "").upper()
        if pil_format == "PNG" or probe.get("has_alpha"):
            return "png"
        return "jpg"

    @staticmethod
    def _probe_image(name: str, data: bytes) -> dict:
        """Validate that ``data`` is a decodable image and read its dimensions."""
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
                Image.DecompressionBombError) as exc:
            raise InvalidRequestError(f"无法解析图片：{name}") from exc
        try:
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                pil_format = image.format or ""
                mode = image.mode
                has_alpha = mode in ("RGBA", "LA") or (
                    mode == "P" and "transparency" in image.info
                )
                orientation = 1
                try:
                    orientation = int(image.getexif().get(0x0112, 1) or 1)
                except Exception:  # noqa: BLE001 - broken EXIF should not fail upload
                    orientation = 1
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidRequestError(f"无法解析图片：{name}") from exc
        if orientation in (5, 6, 7, 8):  # 90/270 degree rotations swap the sides
            width, height = height, width
        return {
            "width": int(width),
            "height": int(height),
            "format": pil_format,
            "mode": mode,
            "has_alpha": bool(has_alpha),
        }

    # -- job id / dirs -----------------------------------------------------

    def _new_job_id(self) -> str:
        with self._lock:
            while True:
                candidate = uuid.uuid4().hex[:12]
                if candidate not in self._jobs:
                    return candidate

    def _remove_job_dirs(self, job_id: str) -> int:
        """Remove all cache folders of a job; returns the number of freed bytes."""
        freed = 0
        for base in (config.UPLOAD_DIR, config.RESULT_DIR, config.TMP_DIR):
            folder = base / job_id
            if not folder.exists():
                continue
            freed += _dir_bytes(folder)
            shutil.rmtree(folder, ignore_errors=True)
        return freed

    def _cleanup_orphan_dirs(self, keep_seconds: float) -> int:
        """Remove folders in the cache dirs that no in-memory job references."""
        freed = 0
        now = datetime.now().timestamp()
        with self._lock:
            known = set(self._jobs)
        for base in (config.UPLOAD_DIR, config.RESULT_DIR, config.TMP_DIR):
            if not base.exists():
                continue
            try:
                children = list(base.iterdir())
            except OSError:
                continue
            for child in children:
                if not child.is_dir() or child.name in known:
                    continue
                try:
                    age = now - child.stat().st_mtime
                except OSError:
                    continue
                if keep_seconds > 0 and age < keep_seconds:
                    continue
                freed += _dir_bytes(child)
                shutil.rmtree(child, ignore_errors=True)
        return freed

    # -- worker ------------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                self._process(job_id)
            except Exception as exc:  # noqa: BLE001 - worker must never die
                self._fail_job(job_id, exc)
            finally:
                self._queue.task_done()

    def _ttl_loop(self) -> None:
        while not self._stop.wait(self._cleanup_interval):
            try:
                self.cleanup_expired()
            except Exception:  # noqa: BLE001 - cleanup must never kill the thread
                continue

    def _process(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.status == "cancelled" or job.token.cancelled:
                self._finalize(job, "cancelled")
                return
            job.status = "running"
            job.processing = True
            if job.started_at is None:
                job.started_at = _now()
        try:
            try:
                engine = get_engine()
            except Exception as exc:  # noqa: BLE001 - report as job error
                message = f"超分引擎初始化失败：{exc}"
                with self._lock:
                    job.error = message
                    for job_file in job.files:
                        if job_file.status in ("pending", "running"):
                            job_file.status = "error"
                            job_file.error = message
                    self._finalize(job, "error")
                return
            with self._lock:
                info = _info_to_dict(getattr(engine, "info", None)) or engine_info_dict()
                job.engine = dict(info)

            for job_file in job.files:
                if job.token.cancelled or job.status == "cancelled":
                    with self._lock:
                        self._cancel_pending(job)
                        self._finalize(job, "cancelled")
                    return
                with self._lock:
                    job_file.status = "running"
                    job_file.progress = 0.0
                    job.current_index = job_file.index
                try:
                    self._process_file(job, job_file, engine)
                except _Cancelled:
                    with self._lock:
                        if job_file.status != "done":
                            job_file.status = "cancelled"
                        self._cancel_pending(job)
                        self._finalize(job, "cancelled")
                    return
                except Exception as exc:  # noqa: BLE001 - per file isolation
                    message = f"处理失败：{exc}" if str(exc) else "处理失败"
                    with self._lock:
                        job_file.status = "error"
                        job_file.error = message
                        job_file.progress = round(min(99.9, job_file.progress), 1)
                        if job.error is None:
                            job.error = message
                    continue
                if job.token.cancelled or job.status == "cancelled":
                    with self._lock:
                        self._discard_result(job_file)
                        job_file.status = "cancelled"
                        self._cancel_pending(job)
                        self._finalize(job, "cancelled")
                    return

            with self._lock:
                if job.status == "cancelled" or job.token.cancelled:
                    self._cancel_pending(job)
                    self._finalize(job, "cancelled")
                elif any(job_file.status == "error" for job_file in job.files):
                    self._finalize(job, "error")
                else:
                    self._finalize(job, "done")
        finally:
            with self._lock:
                job.processing = False

    def _process_file(self, job: _Job, job_file: _JobFile, engine: Any) -> None:
        plan = plan_target(
            job_file.original["width"],
            job_file.original["height"],
            job.target["id"],
            align=job.align,
        )
        image = self._load_image(job_file.upload_path)
        if image.size != (job_file.original["width"], job_file.original["height"]):
            # EXIF orientation handled differently than during probing: re-plan.
            plan = plan_target(image.size[0], image.size[1], job.target["id"], align=job.align)
            job_file.original = {"width": image.size[0], "height": image.size[1]}
        job_file.plan = plan
        out_w = int(plan["output"]["width"])
        out_h = int(plan["output"]["height"])

        requested_model = str(job.options.get("model") or DEFAULT_MODEL)
        requested_level = int(job.options.get("denoise_level") or 0)
        ratio = float(plan.get("scale") or 1.0)
        resolved = self._resolve_model(engine, job, job_file, ratio, requested_model)

        progress_cb = self._make_progress_cb(job, job_file)
        try:
            result = engine.upscale(
                image,
                out_w,
                out_h,
                model=requested_model,
                denoise=bool(job.options.get("denoise")),
                denoise_level=requested_level,
                tile=int(job.options.get("tile") or 0),
                progress_cb=progress_cb,
            )
        except Exception:
            if job.token.cancelled:
                raise _Cancelled() from None
            raise
        if job.token.cancelled:
            raise _Cancelled()
        if not isinstance(result, Image.Image):
            raise RuntimeError("引擎未返回有效图像")
        if result.size != (out_w, out_h):
            # Defensive: the contract says the engine is exact, never ship a mismatch.
            result = result.resize((out_w, out_h), Image.LANCZOS)

        output_format = job_file.output_choice
        stem = sanitize_filename(
            os.path.splitext(job_file.name)[0] or "image", default="image"
        )
        with self._lock:
            candidate = f"{stem}_{job.target['id']}.{output_format}"
            filename = _unique_name(job.result_names, candidate)
            job.result_names.add(filename.casefold())
            result_path = config.RESULT_DIR / job.job_id / filename
        self._save_image(
            result, output_format, int(job.options.get("quality") or config.JPEG_QUALITY),
            result_path,
        )
        note = self._compose_note(engine, job, job_file, resolved)
        with self._lock:
            job_file.result_path = result_path
            job_file.filename = filename
            job_file.output_format = output_format
            try:
                job_file.output_size = result_path.stat().st_size
            except OSError:
                job_file.output_size = None
            job_file.note = note
            job_file.status = "done"
            job_file.progress = 100.0
            job.current_index = job_file.index

    def _resolve_model(
        self,
        engine: Any,
        job: _Job,
        job_file: _JobFile,
        ratio: float,
        requested_model: str,
    ) -> dict | None:
        """Ask the engine which concrete model/weight will be used for this file."""
        resolver = getattr(engine, "resolve_model", None)
        level = int(job.options.get("denoise_level") or 0)
        if callable(resolver):
            try:
                info = resolver(requested_model, ratio, level)
            except Exception as exc:  # noqa: BLE001 - surfaced as a per-file error
                if job.token.cancelled:
                    raise _Cancelled() from None
                # A forced ``pil`` engine may not be able to resolve model weights;
                # that is a graceful compatibility downgrade, not a task failure.
                if str((job.engine or {}).get("backend") or "") == "pil":
                    with self._lock:
                        job_file.model_used = _fallback_model_used(
                            requested_model, ratio
                        )
                    return None
                raise RuntimeError(f"模型解析失败：{exc}") from exc
            if isinstance(info, dict):
                used = info.get("used") or info.get("label") or info.get("id")
                model_used = str(used) if used else _fallback_model_used(requested_model, ratio)
                effective_level = info.get("denoise_level")
                with self._lock:
                    job_file.model_used = model_used
                    if effective_level is not None:
                        try:
                            job_file.denoise_level = int(effective_level)
                        except (TypeError, ValueError):
                            pass
                return info
        model_used = _fallback_model_used(requested_model, ratio)
        with self._lock:
            job_file.model_used = model_used
        return None

    def _compose_note(
        self, engine: Any, job: _Job, job_file: _JobFile, resolved: dict | None
    ) -> str | None:
        """Explain graceful degradations/ignored options in ``files[i].note``."""
        notes: list[str] = []
        requested = str(job.options.get("model") or DEFAULT_MODEL)
        backend = str((job.engine or {}).get("backend") or "")
        is_downscale = str(job_file.plan.get("mode") or "") == "downscale"
        if backend == "pil" and not is_downscale:
            notes.append("AI 模型不可用，已降级为兼容模式（Lanczos 缩放）")
        elif resolved and requested != DEFAULT_MODEL:
            used_id = str(resolved.get("id") or "")
            if used_id and used_id != requested:
                notes.append(f"请求的模型 {requested} 不可用，已降级为 {used_id}")
        family = str((resolved or {}).get("family") or "").lower()
        if not family:
            family = "cugan" if requested.startswith("realcugan") else "rrdb"
        if job.options.get("denoise") and family != "cugan":
            if not bool(getattr(engine, "denoise_available", False)):
                notes.append("降噪不可用，已忽略")
        if requested == "realcugan-conservative" and job.requested_denoise_level > 0:
            notes.append("保守模型不支持降噪等级，已忽略")
        return "；".join(notes) if notes else None

    def _make_progress_cb(self, job: _Job, job_file: _JobFile):
        token = job.token
        lock = self._lock

        def callback(value: Any) -> None:
            if token.cancelled:
                raise _Cancelled()
            try:
                percent = float(value) * 100.0
            except (TypeError, ValueError):
                return
            percent = max(0.0, min(99.9, percent))
            # The callback runs on the worker thread; keep writes tiny and locked.
            with lock:
                if job_file.status == "running":
                    job_file.progress = round(max(job_file.progress, percent), 1)

        return callback

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        try:
            with Image.open(path) as image:
                image.load()
                image = ImageOps.exif_transpose(image)
                mode = image.mode
                if mode in ("RGBA", "LA") or (
                    mode == "P" and "transparency" in image.info
                ):
                    image = image.convert("RGBA")
                elif mode != "RGB":
                    image = image.convert("RGB")
                return image.copy()
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"无法读取上传图片：{exc}") from exc

    @staticmethod
    def _save_image(image: Image.Image, output_format: str, quality: int, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        part_path = path.with_name(path.name + ".part")
        try:
            if output_format == "png":
                image.save(part_path, format="PNG", optimize=True)
            elif output_format == "webp":
                converted = image if image.mode in ("RGB", "RGBA") else image.convert("RGB")
                converted.save(part_path, format="WEBP", quality=int(quality), method=4)
            else:
                _flatten_to_rgb(image).save(
                    part_path, format="JPEG", quality=int(quality),
                    optimize=True, progressive=True,
                )
            os.replace(part_path, path)
        except Exception:
            try:
                part_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @staticmethod
    def _discard_result(job_file: _JobFile) -> None:
        path = job_file.result_path
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        job_file.result_path = None
        job_file.filename = None
        job_file.output_size = None
        job_file.output_format = None

    # -- snapshots (call while holding ``self._lock``) ---------------------

    def _job_progress(self, job: _Job) -> float:
        total_weight = sum(max(1, item.weight) for item in job.files)
        if total_weight <= 0:
            return 0.0
        if job.status == "done":
            return 100.0
        accumulated = 0.0
        for item in job.files:
            weight = max(1, item.weight)
            if item.status == "done":
                percent = 100.0
            elif item.status == "error":
                percent = 100.0
            elif item.status in ("running", "cancelled"):
                percent = max(0.0, min(100.0, item.progress))
            else:
                percent = 0.0
            accumulated += weight * percent
        return round(accumulated / total_weight, 1)

    def _file_public(self, job: _Job, job_file: _JobFile) -> dict:
        data = {
            "index": job_file.index,
            "name": job_file.name,
            "size": job_file.size,
            "original": dict(job_file.original),
            "output": dict(job_file.plan.get("output") or {}),
            "scale": job_file.plan.get("scale"),
            "mode": job_file.plan.get("mode"),
            "aligned": bool(job_file.plan.get("aligned")),
            "model": job_file.model,
            "model_used": job_file.model_used,
            "denoise_level": int(job_file.denoise_level),
            "status": job_file.status,
            "progress": round(float(job_file.progress), 1),
            "output_size": job_file.output_size,
            "output_format": job_file.output_format,
            "note": job_file.note,
            "error": job_file.error,
            "preview_url": f"/api/jobs/{job.job_id}/files/{job_file.index}",
            "download_url": f"/api/jobs/{job.job_id}/files/{job_file.index}/download",
        }
        if job_file.filename:
            data["filename"] = job_file.filename
        return data

    def _job_public(self, job: _Job) -> dict:
        now = _now()
        created = job.created_at
        started = job.started_at
        finished = job.finished_at
        if finished is not None:
            elapsed = (finished - (started or created)).total_seconds()
        else:
            elapsed = (now - (started or created)).total_seconds()
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": self._job_progress(job),
            "current_index": job.current_index,
            "total": len(job.files),
            "created_at": _iso(created),
            "started_at": _iso(started),
            "finished_at": _iso(finished),
            "error": job.error,
            "target": dict(job.target),
            "model": dict(job.model),
            "options": dict(job.options),
            "engine": dict(job.engine) if job.engine else engine_info_dict(),
            "files": [self._file_public(job, item) for item in job.files],
            "elapsed_sec": round(max(0.0, elapsed), 1),
        }

    def _job_summary(self, job: _Job) -> dict:
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": self._job_progress(job),
            "total": len(job.files),
            "created_at": _iso(job.created_at),
        }

    # -- state transitions (call while holding ``self._lock``) -------------

    def _finalize(self, job: _Job, status: str) -> None:
        job.status = status
        job.current_index = None
        if job.finished_at is None:
            job.finished_at = _now()
        if status == "cancelled":
            self._cancel_pending(job)

    @staticmethod
    def _cancel_pending(job: _Job) -> None:
        for job_file in job.files:
            if job_file.status == "pending":
                job_file.status = "cancelled"
                job_file.progress = 0.0

    def _fail_job(self, job_id: str | None, exc: BaseException) -> None:
        if not job_id:
            return
        message = f"任务处理失败：{exc}"
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.error = message
            for job_file in job.files:
                if job_file.status in ("pending", "running"):
                    job_file.status = "error"
                    job_file.error = message
            job.status = "error"
            job.current_index = None
            job.processing = False
            if job.finished_at is None:
                job.finished_at = _now()


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    """Composite transparent pixels onto white before writing JPEG."""
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA"):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def _shutdown_loaded_engine() -> None:
    """Release the GPU model if the engine module was actually imported.

    Kept lazy on purpose: unit tests monkeypatch a fake engine and never import
    torch, so this is a no-op there.
    """
    if "app.engine" not in sys.modules:
        return
    try:
        from .engine import shutdown_engines  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - optional helper
        return
    if callable(shutdown_engines):
        try:
            shutdown_engines()
        except Exception:  # noqa: BLE001 - shutdown is best effort
            pass


# Module level singleton used by ``app.main``.
manager = JobManager()
