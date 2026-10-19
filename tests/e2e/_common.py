"""Shared helpers for the PicPolish E2E suites (Windows + quality checks).

This module intentionally depends only on the runtime stack that the project
venv already provides: ``httpx`` (HTTP client) and ``PIL``/``numpy`` (image
inspection).  It is imported by ``e2e_windows.py`` and ``quality_check.py``.

Nothing here is allowed to start/stop the server: lifecycle is owned by
``e2e_windows.ps1`` so that a failed test can never leave a stray uvicorn.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote

import httpx
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
CACHE_DIR = PROJECT_ROOT / "cache"

# ``docs/API.md`` section 1 / 4: the four tiers, in the documented order.
TIERS_EXPECTED = [
    {"id": "720p", "label": "720P", "rule": "short", "value": 720},
    {"id": "1080p", "label": "1080P", "rule": "short", "value": 1080},
    {"id": "2k", "label": "2K", "rule": "long", "value": 2560},
    {"id": "4k", "label": "4K", "rule": "long", "value": 3840},
]

TIER_LABEL = {t["id"]: t["label"] for t in TIERS_EXPECTED}

# ``docs/API.md`` section 14: fixed model order returned by /api/models.
# Only the parts the docs make unambiguous are asserted exactly; the rest is
# checked structurally so a label/description tweak cannot break acceptance.
MODELS_EXPECTED = [
    {"id": "auto", "family": "rrdb", "denoise_native": False},
    {"id": "realesrgan-x4plus", "family": "rrdb", "denoise_native": False},
    {"id": "realesrgan-x2plus", "family": "rrdb", "denoise_native": False},
    {"id": "realesrgan-anime6b", "family": "rrdb", "denoise_native": False},
    {"id": "realcugan", "family": "cugan", "denoise_native": True},
    {"id": "realcugan-conservative", "family": "cugan", "denoise_native": True},
]
MODEL_IDS = [m["id"] for m in MODELS_EXPECTED]

# Substrings expected in ``files[i].model_used`` for a 1280x720 -> 1080p job
# (ratio 1.5): auto resolves to the 2x model.  Case-insensitive.
MODEL_USED_HINTS = {
    "auto": {"must": ["x2"], "must_not": []},
    "realesrgan-x4plus": {"must": ["x4plus"], "must_not": ["anime"]},
    "realesrgan-x2plus": {"must": ["x2plus"], "must_not": ["anime"]},
    "realesrgan-anime6b": {"must": ["anime"], "must_not": []},
    "realcugan": {"must": ["cugan"], "must_not": ["conservative"]},
    "realcugan-conservative": {"must": ["conservative"], "must_not": []},
}

JPEG_FORMATS = {"JPEG", "MPO"}
MIME_BY_FORMAT = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
EXT_BY_FORMAT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


# ---------------------------------------------------------------------------
# Check registry: every assertion goes through this, so the JSON summary and
# the console output always agree.
# ---------------------------------------------------------------------------
@dataclass
class Check:
    cid: str
    category: str
    status: str  # PASS | FAIL | WARN | SKIP
    detail: str = ""


class Checker:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, cid: str, status: str, detail: str = "", category: str = "") -> "Checker":
        self.checks.append(Check(cid=cid, category=category, status=status, detail=detail))
        marker = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "SKIP": "SKIP"}[status]
        line = f"[{marker}] {cid}"
        if detail:
            line += f"  --  {detail}"
        print(line, flush=True)
        return self

    def pass_(self, cid: str, detail: str = "", category: str = "") -> "Checker":
        return self.add(cid, "PASS", detail, category)

    def fail(self, cid: str, detail: str = "", category: str = "") -> "Checker":
        return self.add(cid, "FAIL", detail, category)

    def warn(self, cid: str, detail: str = "", category: str = "") -> "Checker":
        return self.add(cid, "WARN", detail, category)

    def skip(self, cid: str, detail: str = "", category: str = "") -> "Checker":
        return self.add(cid, "SKIP", detail, category)

    def expect(self, cid: str, ok: bool, detail: str = "", category: str = "") -> bool:
        if ok:
            self.pass_(cid, detail, category)
        else:
            self.fail(cid, detail, category)
        return ok

    def expect_eq(self, cid: str, actual, expected, category: str = "") -> bool:
        ok = actual == expected
        return self.expect(cid, ok, "" if ok else f"expected {expected!r}, got {actual!r}", category)

    def expect_in(self, cid: str, actual, allowed: Iterable, category: str = "") -> bool:
        ok = actual in allowed
        return self.expect(cid, ok, "" if ok else f"expected one of {list(allowed)!r}, got {actual!r}",
                           category)

    def summary(self) -> dict:
        counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
        for check in self.checks:
            counts[check.status] = counts.get(check.status, 0) + 1
        return {
            "total": len(self.checks),
            **{k.lower(): v for k, v in counts.items()},
            "failed_ids": [c.cid for c in self.checks if c.status == "FAIL"],
        }

    def to_json(self) -> list[dict]:
        return [{"id": c.cid, "category": c.category, "status": c.status, "detail": c.detail}
                for c in self.checks]


# ---------------------------------------------------------------------------
# HTTP client for the documented API.
# ---------------------------------------------------------------------------
class ApiError(RuntimeError):
    def __init__(self, message: str, response: httpx.Response | None = None) -> None:
        super().__init__(message)
        self.response = response


class ApiClient:
    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            follow_redirects=True,
            headers={"User-Agent": "PicPolish-QA/1.0"},
        )

    def close(self) -> None:
        self.client.close()

    # -- raw helpers ------------------------------------------------------
    def get(self, path: str, **kwargs) -> httpx.Response:
        return self.client.get(path, **kwargs)

    def post(self, path: str, **kwargs) -> httpx.Response:
        return self.client.post(path, **kwargs)

    def delete(self, path: str, **kwargs) -> httpx.Response:
        return self.client.delete(path, **kwargs)

    # -- documented endpoints --------------------------------------------
    def health(self) -> dict:
        return self._json(self.get("/api/health"))

    def targets(self) -> dict:
        return self._json(self.get("/api/targets"))

    def models(self) -> dict:
        return self._json(self.get("/api/models"))

    def plan(self, images: list[dict], target: str, align: int = 8) -> httpx.Response:
        payload = {"target": target, "align": align,
                   "images": [{"name": i.get("name", ""), "width": i["width"],
                               "height": i["height"]} for i in images]}
        return self.post("/api/plan", json=payload)

    def create_job(self, files: list[Path | tuple[str, bytes, str]], target: str,
                   **form: str) -> httpx.Response:
        parts = []
        for item in files:
            if isinstance(item, tuple):
                name, data, mime = item
                parts.append(("files", (name, data, mime)))
            else:
                path = Path(item)
                parts.append(("files", (path.name, path.read_bytes(), guess_mime(path))))
        data = {"target": target}
        data.update({k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in form.items()})
        return self.post("/api/jobs", files=parts, data=data)

    def job(self, job_id: str) -> dict:
        return self._json(self.get(f"/api/jobs/{job_id}"))

    def cancel_job(self, job_id: str) -> httpx.Response:
        return self.post(f"/api/jobs/{job_id}/cancel")

    def delete_job(self, job_id: str) -> httpx.Response:
        return self.delete(f"/api/jobs/{job_id}")

    def file_bytes(self, job_id: str, index: int, *, download: bool = False,
                   params: dict | None = None) -> httpx.Response:
        suffix = "/download" if download else ""
        return self.get(f"/api/jobs/{job_id}/files/{index}{suffix}", params=params)

    def zip_bytes(self, job_id: str) -> httpx.Response:
        return self.get(f"/api/jobs/{job_id}/download.zip")

    # -- polling ----------------------------------------------------------
    def wait_job(self, job_id: str, *, timeout: float = 300.0, interval: float = 0.4,
                 on_sample=None) -> tuple[dict, float, list[dict]]:
        """Poll ``GET /api/jobs/{id}`` until a terminal status or timeout.

        Returns ``(last_job_dict, elapsed_wall_sec, samples)`` where ``samples``
        is the list of non-terminal snapshots (used to check progress
        monotonicity).  Raises :class:`ApiError` on timeout.
        """
        started = time.monotonic()
        samples: list[dict] = []
        terminal = {"done", "error", "cancelled"}
        while True:
            job = self.job(job_id)
            elapsed = time.monotonic() - started
            if on_sample is not None:
                on_sample(job)
            if job.get("status") in terminal:
                return job, elapsed, samples
            samples.append(job)
            if elapsed > timeout:
                raise ApiError(
                    f"job {job_id} did not finish within {timeout:.0f}s "
                    f"(last status={job.get('status')!r}, progress={job.get('progress')!r})")
            time.sleep(interval)

    def _json(self, response: httpx.Response) -> dict:
        if response.status_code >= 400:
            raise ApiError(f"{response.request.method} {response.request.url} -> "
                           f"{response.status_code}: {response.text[:300]}", response)
        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise ApiError(f"{response.request.url} returned non-JSON: "
                           f"{response.text[:300]}", response) from exc


def guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".bmp": "image/bmp", ".tif": "image/tiff",
        ".tiff": "image/tiff", ".txt": "text/plain",
    }.get(suffix, "application/octet-stream")


# ---------------------------------------------------------------------------
# Image inspection helpers.
# ---------------------------------------------------------------------------
def content_problems(info: dict, reference: dict | None = None) -> list[str]:
    """Hard content-validity gate (Lead + engine-dev mandated).

    A broken tiling pass leaks black tiles, so exact size/format is not enough.
    The plain thresholds are: near-black pixels (RGB max < 10) < 0.5 %, global
    mean > 10 and white saturation < 90 %.

    When the input's stats are supplied as ``reference`` (real photos), the
    *natural* darkness of the source is respected: the result may not have more
    than +2 percentage points of extra black pixels and its mean may not collapse
    below 60 % of the input mean.  A broken tiler still fails loudly (e.g. 92 %
    black vs ~22 % in a night photo).
    """
    problems: list[str] = []
    black_limit = 0.005
    mean_min = 10.0
    if reference is not None:
        black_limit = max(black_limit, float(reference.get("black_ratio", 0.0)) + 0.02)
        mean_min = max(mean_min, float(reference.get("mean", 0.0)) * 0.6)
    if info["mean"] < mean_min:
        problems.append(f"mean={info['mean']:.2f} < {mean_min:.1f}")
    if info["black_ratio"] > black_limit:
        extra = ""
        if reference is not None:
            extra = f", input black={float(reference.get('black_ratio', 0.0)):.2%}"
        problems.append(f"black_ratio={info['black_ratio']:.2%} > {black_limit:.2%} "
                        f"(RGB max<10 pixels{extra}; the tiling bug gave 92%)")
    if info["white_ratio"] > 0.90:
        problems.append(f"white_ratio={info['white_ratio']:.2%} > 90%")
    return problems


def image_info(data: bytes) -> dict:
    """Open image bytes with PIL and report size/format/mode plus pixel stats."""
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        info = {
            "width": im.width,
            "height": im.height,
            "size": im.size,
            "format": (im.format or "").upper(),
            "mode": im.mode,
            "bytes": len(data),
        }
        rgba = im.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    # Composite over neutral grey so fully transparent regions (e.g. the RGBA
    # sample's hole) are not mistaken for black content.
    alpha01 = (alpha / 255.0)[..., None]
    rgb_comp = rgb * alpha01 + 128.0 * (1.0 - alpha01)
    gray = rgb_comp.mean(axis=2)
    channel_max = rgb_comp.max(axis=2)
    channel_min = rgb_comp.min(axis=2)
    info.update({
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "min": float(gray.min()),
        "max": float(gray.max()),
        "channel_means": [float(x) for x in rgb_comp.reshape(-1, 3).mean(axis=0)],
        # Content-validity metric (Lead + engine-dev): near-black pixels are
        # RGB max < 10; a broken tiler shows 90%+ black, healthy photos < 0.1%.
        "black_ratio": float((channel_max < 10).mean()),
        "white_ratio": float((channel_min > 247).mean()),
        # Legacy aliases (same thresholds now).
        "near_black_ratio": float((channel_max < 10).mean()),
        "near_white_ratio": float((channel_min > 247).mean()),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "alpha_transparent_ratio": float((alpha <= 32).mean()),
    })
    return info


def laplacian_variance(im: Image.Image) -> float:
    """Variance of the 4-neighbour Laplacian response on the grayscale image."""
    gray = np.asarray(im.convert("L"), dtype=np.float64)
    lap = (4.0 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1]
           - gray[1:-1, :-2] - gray[1:-1, 2:])
    return float(lap.var())


# ---------------------------------------------------------------------------
# Model-matrix helpers: image difference metrics + model_used matching.
# ---------------------------------------------------------------------------
def _rgb_array(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), dtype=np.float64)


def mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    """Mean absolute per-channel difference between two equally sized images."""
    if a.size != b.size:
        raise ValueError(f"size mismatch: {a.size} vs {b.size}")
    return float(np.abs(_rgb_array(a) - _rgb_array(b)).mean())


def rmse(a: Image.Image, b: Image.Image) -> float:
    if a.size != b.size:
        raise ValueError(f"size mismatch: {a.size} vs {b.size}")
    diff = _rgb_array(a) - _rgb_array(b)
    return float(np.sqrt((diff ** 2).mean()))


def differing_pixel_ratio(a: Image.Image, b: Image.Image, threshold: float = 2.0) -> float:
    """Fraction of pixels whose max channel difference exceeds ``threshold``."""
    if a.size != b.size:
        raise ValueError(f"size mismatch: {a.size} vs {b.size}")
    diff = np.abs(_rgb_array(a) - _rgb_array(b))
    return float((diff.max(axis=2) > threshold).mean())


def model_used_matches(model_id: str, model_used: str | None,
                       hints: dict | None = None) -> tuple[bool, str]:
    """Tolerant check that ``model_used`` really resolved to the requested model.

    ``hints`` overrides the default ``MODEL_USED_HINTS[model_id]`` when the
    expected family depends on the upscale ratio (e.g. ``auto`` picks x2plus at
    ratio <= 2 but x4plus at ratio > 2).
    """
    if not isinstance(model_used, str) or not model_used.strip():
        return False, f"model_used={model_used!r} is empty"
    used = model_used.lower()
    effective = hints if hints is not None else MODEL_USED_HINTS.get(model_id, {})
    must = list(effective.get("must", []))
    must_not = list(effective.get("must_not", []))
    missing = [token for token in must if token not in used]
    forbidden = [token for token in must_not if token in used]
    if missing:
        return False, f"model_used={model_used!r} missing {missing}"
    if forbidden:
        return False, f"model_used={model_used!r} unexpectedly contains {forbidden}"
    return True, f"model_used={model_used!r}"


def image_stats(im: Image.Image) -> dict:
    return image_info(image_to_bytes(im))


def image_to_bytes(im: Image.Image, fmt: str = "PNG", quality: int = 95) -> bytes:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        save_im = im.convert("RGB")
        save_im.save(buf, format="JPEG", quality=quality)
    else:
        save_im = im.convert("RGBA") if im.mode == "RGBA" else im.convert("RGB")
        save_im.save(buf, format="PNG")
    return buf.getvalue()


def resample_baseline(im: Image.Image, size: tuple[int, int],
                      resample: int = Image.LANCZOS) -> Image.Image:
    """Plain resampler baseline (Lanczos by default) for quality comparisons."""
    return im.convert("RGB").resize(size, resample)


def bicubic_baseline(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Bicubic upscale baseline (kept for reporting next to Lanczos)."""
    return resample_baseline(im, size, Image.BICUBIC)


def tile_metrics(im: Image.Image, grid: int = 4, block: int = 32) -> dict:
    """Uniformity / seam metrics used to catch broken-tiling outputs.

    On a near-uniform gradient every ``grid``x``grid`` cell and every
    ``block``x``block`` pixel block must have almost the global mean, and the
    column/row mean profiles must be smooth.  A black tile or a seam line
    produces a large deviation / jump.
    """
    gray = np.asarray(im.convert("RGB"), dtype=np.float64).mean(axis=2)
    height, width = gray.shape
    global_mean = float(gray.mean())
    cell_means: list[float] = []
    cell_h = max(1, height // grid)
    cell_w = max(1, width // grid)
    for gy in range(grid):
        for gx in range(grid):
            cell = gray[gy * cell_h:(gy + 1) * cell_h, gx * cell_w:(gx + 1) * cell_w]
            cell_means.append(float(cell.mean()))
    block_means: list[float] = []
    for y0 in range(0, max(1, height - block + 1), block):
        for x0 in range(0, max(1, width - block + 1), block):
            block_means.append(float(gray[y0:y0 + block, x0:x0 + block].mean()))
    col_diffs = np.abs(np.diff(gray.mean(axis=0)))
    row_diffs = np.abs(np.diff(gray.mean(axis=1)))
    return {
        "global_mean": global_mean,
        "cell_means": cell_means,
        "max_cell_deviation": max(abs(m - global_mean) for m in cell_means),
        "block_count": len(block_means),
        "max_block_deviation": max(abs(m - global_mean) for m in block_means),
        "column_jump_max": float(col_diffs.max()) if col_diffs.size else 0.0,
        "row_jump_max": float(row_diffs.max()) if row_diffs.size else 0.0,
        "column_jump_median": float(np.median(col_diffs)) if col_diffs.size else 0.0,
        "row_jump_median": float(np.median(row_diffs)) if row_diffs.size else 0.0,
    }


# ---------------------------------------------------------------------------
# Content-Disposition / filename helpers.
# ---------------------------------------------------------------------------
def parse_content_disposition(value: str | None) -> dict:
    if not value:
        return {}
    params: dict[str, str] = {}
    parts = value.split(";")
    params["_type"] = parts[0].strip().lower()
    for part in parts[1:]:
        part = part.strip()
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        key = key.strip().lower()
        raw = raw.strip()
        if len(raw) >= 2 and raw[0] == raw[-1] == '"':
            raw = raw[1:-1]
        params[key] = raw
    filename = params.get("filename*")
    if filename:
        # RFC 5987: charset'lang'percent-encoded-value
        if "''" in filename:
            filename = filename.split("''", 1)[1]
        params["filename_decoded"] = unquote(filename)
    elif "filename" in params:
        params["filename_decoded"] = params["filename"]
    return params


def content_disposition_filename(value: str | None) -> str | None:
    params = parse_content_disposition(value)
    return params.get("filename_decoded") or params.get("filename")


# ---------------------------------------------------------------------------
# Misc helpers.
# ---------------------------------------------------------------------------
def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cache_contents() -> dict:
    """File listing of cache/uploads|results|tmp (used by lifecycle checks)."""
    result: dict[str, list[str]] = {}
    for name in ("uploads", "results", "tmp"):
        folder = CACHE_DIR / name
        files = []
        if folder.is_dir():
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    files.append(str(path.relative_to(CACHE_DIR)))
        result[name] = files
    return result


def cache_has_files() -> dict:
    return {name: len(files) for name, files in cache_contents().items()}


def setup_console() -> None:
    """Make sure non-ASCII detail strings never crash a cp936 console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def approx(actual: float, expected: float, tol: float = 1e-6) -> bool:
    return abs(float(actual) - float(expected)) <= tol


def fmt_sec(value: float | None) -> str:
    if value is None:
        return "?"
    return f"{value:.2f}s"


def load_samples(names: Iterable[str]) -> list[Path]:
    paths = []
    for name in names:
        path = SAMPLES_DIR / name
        if not path.is_file():
            raise SystemExit(f"missing sample {path} -- run tests/e2e/make_samples.py first")
        paths.append(path)
    return paths


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")
