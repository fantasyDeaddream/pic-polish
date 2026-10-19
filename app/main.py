"""FastAPI application for PicPolish (routes per ``docs/API.md``).

Run with::

    .venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8799

The module stays import-light: torch / the Real-ESRGAN engine are loaded lazily
by the job worker (``app.jobs``) so ``GET /api/health`` answers quickly and
without touching the GPU.
"""
from __future__ import annotations

import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config, jobs
from .target import TIERS, InvalidTarget, get_tier, plan_many

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


def _cleanup_tmp_at_startup() -> None:
    """Remove stale intermediate files from ``cache/tmp`` (always transient)."""
    folder = config.TMP_DIR
    if not folder.exists():
        return
    try:
        children = list(folder.iterdir())
    except OSError:
        return
    for child in children:
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            continue


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.ensure_dirs()
    _cleanup_tmp_at_startup()
    try:
        yield
    finally:
        try:
            jobs.manager.shutdown()
        except Exception:  # noqa: BLE001 - shutdown must never crash the server
            pass
        if config.CLEAN_ON_EXIT:
            try:
                config.clear_cache_images()
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass


app = FastAPI(
    title=config.APP_NAME,
    version=config.VERSION,
    description="PicPolish image super-resolution API",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# JSON error mapping (Chinese, human readable details)
# ---------------------------------------------------------------------------


@app.exception_handler(jobs.JobError)
async def _job_error_handler(_request: Request, exc: jobs.JobError) -> JSONResponse:
    return JSONResponse(status_code=getattr(exc, "status_code", 400), content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    detail = "请求参数格式错误"
    errors = exc.errors() or []
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.get("loc", ()))
        message = first.get("msg", "")
        detail = f"请求参数格式错误：{location} {message}".strip()
    return JSONResponse(status_code=400, content={"detail": detail})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_int(raw: Any, name: str, default: int, low: int, high: int) -> int:
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        raise HTTPException(status_code=400, detail=f"{name} 必须是 {low}..{high} 的整数")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail=f"{name} 必须是 {low}..{high} 的整数"
        ) from exc
    if isinstance(raw, float) and raw != value:
        raise HTTPException(status_code=400, detail=f"{name} 必须是 {low}..{high} 的整数")
    if not (low <= value <= high):
        raise HTTPException(status_code=400, detail=f"{name} 必须是 {low}..{high} 的整数")
    return value


def _parse_bool(raw: Any, name: str, default: bool = False) -> bool:
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise HTTPException(status_code=400, detail=f"{name} 必须是布尔值（true/false）")


def _coerce_positive_int(value: Any, index: int, field: str) -> int:
    label = f"第 {index + 1} 张图片的 {field}"
    if isinstance(value, bool) or value is None:
        raise HTTPException(status_code=400, detail=f"{label} 必须是正整数")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{label} 必须是正整数") from exc
    if isinstance(value, float) and value != number:
        raise HTTPException(status_code=400, detail=f"{label} 必须是正整数")
    if number <= 0:
        raise HTTPException(status_code=400, detail=f"{label} 必须是正整数")
    return number


def _file_headers(content_type: str | None, attached_name: str | None = None) -> dict[str, str]:
    headers = {"Cache-Control": "no-cache"}
    if attached_name:
        headers["Content-Disposition"] = jobs.content_disposition(attached_name)
    return headers


def _require_file(job_id: str, index: int) -> tuple[dict, dict]:
    """Return ``(job_snapshot, file_snapshot)`` or raise 404/409."""
    snapshot = jobs.manager.get(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"任务不存在：{job_id}")
    if index < 0 or index >= len(snapshot["files"]):
        raise HTTPException(status_code=404, detail=f"文件不存在：index={index}")
    file_snapshot = snapshot["files"][index]
    if file_snapshot.get("status") != "done":
        raise HTTPException(status_code=409, detail="结果尚未生成")
    return snapshot, file_snapshot


# ---------------------------------------------------------------------------
# API: meta
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def api_health() -> dict:
    engine = jobs.engine_info_dict()
    stats = jobs.manager.stats()
    return {
        "status": "ok",
        "app": config.APP_NAME,
        "version": config.VERSION,
        "engine": engine,
        "models": jobs.models_status(),
        "limits": {
            "max_files": config.MAX_FILES,
            "max_file_mb": config.MAX_FILE_MB,
            "max_total_mb": config.MAX_TOTAL_MB,
            "max_pixels": config.MAX_PIXELS,
        },
        "cache": {
            "jobs": int(stats.get("jobs") or 0),
            "result_mb": round(float(stats.get("result_bytes") or 0) / (1024 * 1024), 1),
        },
    }


@app.get("/api/targets")
async def api_targets() -> dict:
    return {"targets": [dict(tier) for tier in TIERS.values()]}


@app.get("/api/models")
async def api_models() -> dict:
    """Model registry (docs/API.md §14.1); never loads torch or model weights."""
    return jobs.list_models()


@app.post("/api/plan")
async def api_plan(payload: Annotated[dict, Body()]) -> dict:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象")
    target = payload.get("target")
    if target is None or (isinstance(target, str) and not target.strip()):
        raise HTTPException(
            status_code=400, detail="缺少 target（可选 720p/1080p/2k/4k）"
        )
    align = _parse_int(
        payload.get("align", config.DEFAULT_ALIGN),
        "align",
        config.DEFAULT_ALIGN,
        config.MIN_ALIGN,
        config.MAX_ALIGN,
    )
    try:
        tier = get_tier(target.strip() if isinstance(target, str) else target)
    except InvalidTarget as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise HTTPException(status_code=400, detail="images 必须是非空数组")
    if len(images) > 200:
        raise HTTPException(status_code=400, detail="一次最多计算 200 张图片")

    normalized: list[dict] = []
    for index, item in enumerate(images):
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=400, detail=f"第 {index + 1} 张图片信息格式错误（应为对象）"
            )
        width = _coerce_positive_int(item.get("width"), index, "width")
        height = _coerce_positive_int(item.get("height"), index, "height")
        name = str(item.get("name") or f"image_{index + 1}")
        normalized.append({"name": name, "width": width, "height": height})

    try:
        plans = plan_many(normalized, tier["id"], align=align)
    except InvalidTarget as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response_images = [
        {
            "index": plan["index"],
            "name": plan["name"],
            "origin": plan["origin"],
            "output": plan["output"],
            "scale": plan["scale"],
            "mode": plan["mode"],
            "aligned": plan["aligned"],
            "note": plan["note"],
        }
        for plan in plans
    ]
    return {
        "target": {
            "id": tier["id"],
            "label": tier["label"],
            "rule": tier["rule"],
            "value": tier["value"],
            "align": align,
        },
        "images": response_images,
    }


# ---------------------------------------------------------------------------
# API: jobs
# ---------------------------------------------------------------------------


@app.post("/api/jobs", status_code=202)
async def api_create_job(
    files: Annotated[list[UploadFile] | None, File()] = None,
    target: Annotated[str | None, Form()] = None,
    align: Annotated[str | None, Form()] = None,
    model: Annotated[str | None, Form()] = None,
    denoise: Annotated[str | None, Form()] = None,
    denoise_level: Annotated[str | None, Form()] = None,
    fmt: Annotated[str | None, Form(alias="format")] = None,
    quality: Annotated[str | None, Form()] = None,
    tile: Annotated[str | None, Form()] = None,
) -> dict:
    if not target or not str(target).strip():
        raise HTTPException(
            status_code=400, detail="缺少 target 参数（可选 720p/1080p/2k/4k）"
        )
    uploads = files or []
    if not uploads:
        raise HTTPException(status_code=400, detail="请至少上传 1 张图片")
    if len(uploads) > config.MAX_FILES:
        raise HTTPException(status_code=400, detail=f"一次最多 {config.MAX_FILES} 张图片")

    options = {
        "align": _parse_int(
            align, "align", config.DEFAULT_ALIGN, config.MIN_ALIGN, config.MAX_ALIGN
        ),
        "model": "auto" if model is None or not str(model).strip() else str(model).strip(),
        "denoise": _parse_bool(denoise, "denoise", False),
        "denoise_level": _parse_int(denoise_level, "denoise_level", 0, 0, 3),
        "format": "auto" if fmt is None or not str(fmt).strip() else str(fmt).strip(),
        "quality": _parse_int(quality, "quality", config.JPEG_QUALITY, 70, 100),
        "tile": _parse_int(tile, "tile", 0, 0, 1024),
    }
    if options["tile"] not in (0,) and options["tile"] < 64:
        raise HTTPException(status_code=400, detail="tile 必须是 0 或 64..1024 的整数")

    max_read = max(1, config.MAX_FILE_MB) * 1024 * 1024
    payloads: list[tuple[str, bytes]] = []
    for upload in uploads:
        raw_name = upload.filename or "image"
        chunks: list[bytes] = []
        received = 0
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            received += len(chunk)
            if received > max_read:
                raise HTTPException(
                    status_code=413,
                    detail=f"单个文件超过 {config.MAX_FILE_MB}MB 限制：{raw_name}",
                )
            chunks.append(chunk)
        payloads.append((raw_name, b"".join(chunks)))
        try:
            await upload.close()
        except Exception:  # noqa: BLE001 - closing is best effort
            pass

    return jobs.manager.submit(payloads, str(target).strip(), options)


@app.get("/api/jobs/{job_id}")
async def api_get_job(job_id: str) -> dict:
    snapshot = jobs.manager.get(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"任务不存在：{job_id}")
    return snapshot


@app.get("/api/jobs/{job_id}/files/{index}")
async def api_preview_file(job_id: str, index: int) -> FileResponse:
    _snapshot, file_snapshot = _require_file(job_id, index)
    path = jobs.manager.result_path(job_id, index)
    if path is None or not path.is_file():
        raise HTTPException(status_code=409, detail="结果尚未生成")
    content_type = jobs.IMAGE_CONTENT_TYPES.get(
        str(file_snapshot.get("output_format") or ""), "application/octet-stream"
    )
    return FileResponse(path, media_type=content_type, headers=_file_headers(content_type))


@app.get("/api/jobs/{job_id}/files/{index}/download")
async def api_download_file(
    job_id: str, index: int, filename: str | None = None
) -> FileResponse:
    snapshot, file_snapshot = _require_file(job_id, index)
    path = jobs.manager.result_path(job_id, index)
    if path is None or not path.is_file():
        raise HTTPException(status_code=409, detail="结果尚未生成")
    output_format = str(file_snapshot.get("output_format") or "jpg")
    extension = "jpg" if output_format in ("jpg", "jpeg") else output_format
    default_name = str(
        file_snapshot.get("filename")
        or f"{job_id}_{index}.{extension}"
    )
    attached_name = jobs.sanitize_download_filename(filename, extension) or default_name
    content_type = jobs.IMAGE_CONTENT_TYPES.get(output_format, "application/octet-stream")
    return FileResponse(
        path,
        media_type=content_type,
        headers=_file_headers(content_type, attached_name),
    )


@app.get("/api/jobs/{job_id}/download.zip")
async def api_download_zip(job_id: str) -> Response:
    data, archive_name = jobs.manager.build_zip(job_id)
    return Response(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{archive_name}"',
            "Cache-Control": "no-cache",
        },
    )


@app.post("/api/jobs/{job_id}/cancel")
async def api_cancel_job(job_id: str) -> dict:
    return jobs.manager.cancel(job_id)


@app.delete("/api/jobs/{job_id}")
async def api_delete_job(job_id: str) -> dict:
    return jobs.manager.delete(job_id)


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

try:
    app.mount(
        "/web",
        StaticFiles(directory=str(config.WEB_DIR), html=True, check_dir=False),
        name="web",
    )
except Exception:  # pragma: no cover - static hosting is best effort
    pass


@app.get("/", include_in_schema=False)
async def index_page() -> FileResponse:
    index_path = Path(config.WEB_DIR) / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="网页资源尚未就绪：web/index.html 不存在")
    return FileResponse(index_path, media_type="text/html; charset=utf-8")


if __name__ == "__main__":  # pragma: no cover - manual debugging helper
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
    )
