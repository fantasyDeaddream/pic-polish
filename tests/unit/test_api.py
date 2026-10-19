"""Unit tests for the PicPolish HTTP API and job queue.

Everything runs against a *fake* engine (monkeypatched into ``app.jobs``) so the
suite is fast, never touches torch/CUDA and needs no model weights:

    .venv\\Scripts\\python.exe -m pytest tests\\unit\\test_api.py -q
"""
from __future__ import annotations

import io
import random
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, jobs
from app import main as main_module
from app.target import plan_target


# ---------------------------------------------------------------------------
# fake engine + helpers
# ---------------------------------------------------------------------------

MODEL_IDS = [
    "auto",
    "realesrgan-x4plus",
    "realesrgan-x2plus",
    "realesrgan-anime6b",
    "realcugan",
    "realcugan-conservative",
]

_MODEL_META = (
    (
        "auto", "智能推荐", "rrdb", [2, 4],
        {"supported": True, "native": False, "levels": [0, 1], "default": 0},
        "BSD-3-Clause", 127.9,
        "按目标倍率自动选择 Real-ESRGAN x2plus / x4plus",
        ["通用"], ["不确定选哪个时"],
        "https://github.com/xinntao/Real-ESRGAN",
    ),
    (
        "realesrgan-x4plus", "Real-ESRGAN 通用照片", "rrdb", [4],
        {"supported": True, "native": False, "levels": [0, 1], "default": 0},
        "BSD-3-Clause", 63.9,
        "通用照片、风景、人像超分",
        ["照片", "通用"], ["照片、风景、人像"],
        "https://github.com/xinntao/Real-ESRGAN",
    ),
    (
        "realesrgan-x2plus", "Real-ESRGAN 低倍率", "rrdb", [2],
        {"supported": True, "native": False, "levels": [0, 1], "default": 0},
        "BSD-3-Clause", 64.0,
        "只放大 ≤2 倍时更自然",
        ["照片", "低倍率"], ["只需要 ≤2 倍放大"],
        "https://github.com/xinntao/Real-ESRGAN",
    ),
    (
        "realesrgan-anime6b", "Real-ESRGAN 动漫", "rrdb", [4],
        {"supported": True, "native": False, "levels": [0, 1], "default": 0},
        "BSD-3-Clause", 17.1,
        "动漫截图 / 插画专用",
        ["动漫", "插画"], ["动漫截图、插画、二次元"],
        "https://github.com/xinntao/Real-ESRGAN",
    ),
    (
        "realcugan", "Real-CUGAN（动漫/视频）", "cugan", [2, 3, 4],
        {"supported": True, "native": True, "levels": [0, 1, 2, 3], "default": 0},
        "MIT", 55.4,
        "Real-CUGAN v3，原生降噪，适合动漫/视频截图",
        ["动漫", "降噪", "视频"], ["噪点明显、需要降噪"],
        "https://github.com/bilibili/ailab",
    ),
    (
        "realcugan-conservative", "Real-CUGAN 保守", "cugan", [2, 3, 4],
        {"supported": False, "native": True, "levels": [0], "default": 0},
        "MIT", 15.2,
        "保守权重，减少过度锐化/风格化",
        ["动漫", "保守"], ["不想过度锐化"],
        "https://github.com/bilibili/ailab",
    ),
)


def make_models_payload(available: set[str] | None = None) -> dict:
    """Return a registry payload shaped exactly like ``app.models.list_models()``."""
    models = []
    for (model_id, label, family, scales, denoise, license_name, size_mb,
         description, tags, recommended, source) in _MODEL_META:
        models.append(
            {
                "id": model_id,
                "label": label,
                "family": family,
                "native_scales": list(scales),
                "description": description,
                "tags": list(tags),
                "recommended_for": list(recommended),
                "denoise": dict(denoise),
                "license": license_name,
                "size_mb": size_mb,
                "source": source,
                "available": True if available is None else model_id in available,
            }
        )
    return {"default": "auto", "models": models}


class FakeEngine:
    """Deterministic PIL based stand-in for ``app.engine.BaseEngine``."""

    def __init__(
        self,
        *,
        fail: bool = False,
        wrong_size: bool = False,
        gate: threading.Event | None = None,
        steps: int = 5,
        loop_delay: float = 0.0,
        backend: str = "realesrgan",
        denoise_available: bool = False,
    ) -> None:
        is_ai = backend == "realesrgan"
        self.info = SimpleNamespace(
            backend=backend,
            device="cuda" if is_ai else "cpu",
            gpu="Fake GPU" if is_ai else None,
            model="RealESRGAN_x4plus" if is_ai else None,
            native_scale=4 if is_ai else 1,
            models_available=tuple(MODEL_IDS) if is_ai else (),
        )
        self.fail = fail
        self.wrong_size = wrong_size
        self.gate = gate
        self.steps = steps
        self.loop_delay = loop_delay
        self.denoise_available = denoise_available
        self.calls: list[dict] = []
        self.resolve_calls: list[tuple] = []

    def resolve_model(self, model: str, ratio: float, denoise_level: int = 0) -> dict:
        self.resolve_calls.append((model, float(ratio), int(denoise_level)))
        if model == "auto":
            if ratio <= 2.0:
                return {
                    "id": "realesrgan-x2plus", "label": "Real-ESRGAN 低倍率",
                    "used": "RealESRGAN_x2plus", "family": "rrdb", "native_scale": 2,
                    "weight_file": "models/RealESRGAN_x2plus.pth",
                    "denoise_level": denoise_level,
                }
            return {
                "id": "realesrgan-x4plus", "label": "Real-ESRGAN 通用照片",
                "used": "RealESRGAN_x4plus", "family": "rrdb", "native_scale": 4,
                "weight_file": "models/RealESRGAN_x4plus.pth",
                "denoise_level": denoise_level,
            }
        if model == "realcugan":
            scale = 2 if ratio <= 2.0 else 3 if ratio <= 3.0 else 4
            level = denoise_level if scale == 2 else (0 if denoise_level == 0 else 3)
            return {
                "id": "realcugan", "label": "Real-CUGAN（动漫/视频）",
                "used": f"Real-CUGAN {scale}x", "family": "cugan",
                "native_scale": scale, "weight_file": f"models/realcugan/up{scale}x.pth",
                "denoise_level": level,
            }
        if model == "realcugan-conservative":
            return {
                "id": model, "label": "Real-CUGAN 保守",
                "used": "Real-CUGAN conservative", "family": "cugan",
                "native_scale": 2, "weight_file": "models/realcugan/up2x-conservative.pth",
                "denoise_level": 0,
            }
        used = {
            "realesrgan-x4plus": "RealESRGAN_x4plus",
            "realesrgan-x2plus": "RealESRGAN_x2plus",
            "realesrgan-anime6b": "RealESRGAN_x4plus_anime_6B",
        }.get(model, model)
        return {
            "id": model, "label": used, "used": used, "family": "rrdb",
            "native_scale": 2 if model.endswith("x2plus") else 4,
            "weight_file": f"models/{used}.pth", "denoise_level": denoise_level,
        }

    def upscale(
        self, img, out_w, out_h, *, model="auto", denoise=False,
        denoise_level=0, tile=0, progress_cb=None,
    ):
        self.calls.append(
            {
                "size": img.size,
                "out": (out_w, out_h),
                "model": model,
                "denoise": denoise,
                "denoise_level": denoise_level,
                "tile": tile,
            }
        )
        if self.gate is not None:
            self.gate.wait(timeout=10.0)
        for step in range(1, self.steps + 1):
            if progress_cb is not None:
                progress_cb(step / self.steps)
            if self.loop_delay:
                time.sleep(self.loop_delay)
        if self.fail:
            raise RuntimeError("fake engine boom")
        if self.wrong_size:
            return img.resize((out_w + 2, out_h + 2), Image.LANCZOS)
        return img.resize((out_w, out_h), Image.LANCZOS)

    def close(self) -> None:
        pass


def image_bytes(
    fmt: str = "PNG",
    size: tuple[int, int] = (320, 240),
    color: tuple[int, ...] = (200, 40, 40),
    mode: str = "RGB",
) -> bytes:
    fill = (*color[:3], 128) if mode == "RGBA" else color
    buffer = io.BytesIO()
    Image.new(mode, size, fill).save(buffer, fmt)
    return buffer.getvalue()


def noise_image_bytes(size: tuple[int, int] = (320, 240), fmt: str = "PNG") -> bytes:
    rng = random.Random(1234)
    payload = bytes(rng.randrange(256) for _ in range(size[0] * size[1] * 3))
    image = Image.frombytes("RGB", size, payload)
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


def submit(client: TestClient, images, target: str = "720p", **form):
    files = [("files", (name, data, "application/octet-stream")) for name, data in images]
    data = {"target": target}
    data.update({key: str(value) for key, value in form.items()})
    return client.post("/api/jobs", files=files, data=data)


def _get_job(client: TestClient, job_id: str) -> dict:
    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200, response.text
    return response.json()


def _wait_for(predicate, timeout: float = 8.0, interval: float = 0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition was not met in time")


def wait_finished(client: TestClient, job_id: str, timeout: float = 8.0) -> dict:
    def finished():
        data = _get_job(client, job_id)
        return data if data["status"] in ("done", "error", "cancelled") else None

    return _wait_for(finished, timeout=timeout)


def wait_status(client: TestClient, job_id: str, status: str, timeout: float = 8.0) -> dict:
    def matches():
        data = _get_job(client, job_id)
        return data if data["status"] == status else None

    return _wait_for(matches, timeout=timeout)


def wait_file_status(
    client: TestClient, job_id: str, index: int, status: str, timeout: float = 8.0
) -> dict:
    def matches():
        data = _get_job(client, job_id)
        item = data["files"][index]
        return item if item["status"] == status else None

    return _wait_for(matches, timeout=timeout)


def submit_and_wait(client: TestClient, images, target: str = "720p", **form):
    response = submit(client, images, target=target, **form)
    assert response.status_code == 202, response.text
    created = response.json()
    return created, wait_finished(client, created["job_id"])


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def env(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(config, "CACHE_DIR", cache_root)
    monkeypatch.setattr(config, "UPLOAD_DIR", cache_root / "uploads")
    monkeypatch.setattr(config, "RESULT_DIR", cache_root / "results")
    monkeypatch.setattr(config, "TMP_DIR", cache_root / "tmp")
    monkeypatch.setattr(config, "CLEAN_ON_EXIT", False)

    engine = FakeEngine()
    monkeypatch.setattr(jobs, "get_engine", lambda force=None: engine)
    engine_info = {
        "backend": "realesrgan",
        "device": "cuda",
        "gpu": "Fake GPU",
        "model": "RealESRGAN_x4plus",
        "native_scale": 4,
    }
    monkeypatch.setattr(jobs, "engine_info_dict", lambda refresh=False: dict(engine_info))
    models_payload = make_models_payload()
    monkeypatch.setattr(jobs, "list_models", lambda: models_payload)

    manager = jobs.JobManager(cleanup_interval=3600.0)
    monkeypatch.setattr(jobs, "manager", manager)

    with TestClient(main_module.app) as client:
        yield SimpleNamespace(
            client=client,
            manager=manager,
            engine=engine,
            engine_info=engine_info,
            models_payload=models_payload,
            cache_root=cache_root,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    manager.shutdown()


# ---------------------------------------------------------------------------
# meta endpoints
# ---------------------------------------------------------------------------


def test_health(env):
    started = time.monotonic()
    response = env.client.get("/api/health")
    elapsed = time.monotonic() - started
    assert response.status_code == 200
    assert elapsed < 2.0, "health check must answer fast (no GPU init)"
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == config.APP_NAME
    assert data["version"] == config.VERSION
    assert data["engine"] == env.engine_info
    expected_available = [
        item["id"] for item in env.models_payload["models"] if item["available"]
    ]
    assert data["models"] == {"default": "auto", "available": expected_available}
    assert data["limits"] == {
        "max_files": config.MAX_FILES,
        "max_file_mb": config.MAX_FILE_MB,
        "max_total_mb": config.MAX_TOTAL_MB,
        "max_pixels": config.MAX_PIXELS,
    }
    assert set(data["cache"]) == {"jobs", "result_mb"}
    assert isinstance(data["cache"]["jobs"], int)
    assert isinstance(data["cache"]["result_mb"], float)


def test_targets(env):
    response = env.client.get("/api/targets")
    assert response.status_code == 200
    targets = response.json()["targets"]
    assert [item["id"] for item in targets] == ["720p", "1080p", "2k", "4k"]
    for item in targets:
        assert set(item) == {"id", "label", "rule", "value", "hint"}
        assert item["rule"] in ("short", "long")
        assert item["hint"]


def test_models_endpoint(env):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("/api/models must not initialize the upscale engine")

    env.monkeypatch.setattr(jobs, "get_engine", forbidden)
    started = time.monotonic()
    response = env.client.get("/api/models")
    elapsed = time.monotonic() - started
    assert response.status_code == 200
    assert elapsed < 2.0, "/api/models must answer immediately (no model loading)"
    data = response.json()
    assert data["default"] == "auto"
    assert [model["id"] for model in data["models"]] == MODEL_IDS
    required = {
        "id", "label", "description", "tags", "recommended_for", "family",
        "native_scales", "denoise", "license", "size_mb", "source", "available",
    }
    for model in data["models"]:
        assert required <= set(model)
        assert model["family"] in ("rrdb", "cugan")
        assert isinstance(model["native_scales"], list)
        assert {"supported", "native", "levels", "default"} <= set(model["denoise"])
        assert isinstance(model["available"], bool)
    assert all(model["available"] is True for model in data["models"])


def test_models_endpoint_reports_unavailable(env):
    payload = make_models_payload(
        available={
            "auto", "realesrgan-x4plus", "realesrgan-x2plus", "realesrgan-anime6b",
        }
    )
    env.monkeypatch.setattr(jobs, "list_models", lambda: payload)
    models = {
        model["id"]: model for model in env.client.get("/api/models").json()["models"]
    }
    assert models["realcugan"]["available"] is False
    assert models["realcugan-conservative"]["available"] is False
    assert models["auto"]["available"] is True
    health = env.client.get("/api/health").json()
    assert health["models"] == {
        "default": "auto",
        "available": [
            "auto", "realesrgan-x4plus", "realesrgan-x2plus", "realesrgan-anime6b",
        ],
    }


# ---------------------------------------------------------------------------
# /api/plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "height", "tier", "expected", "mode"),
    [
        (1280, 720, "1080p", (1920, 1080), "upscale"),
        (1000, 667, "1080p", (1616, 1080), "upscale"),
        (1080, 1920, "720p", (720, 1280), "downscale"),
        (3840, 2160, "1080p", (1920, 1080), "downscale"),
        (1920, 1080, "2k", (2560, 1440), "upscale"),
    ],
)
def test_plan_examples(env, width, height, tier, expected, mode):
    response = env.client.post(
        "/api/plan",
        json={
            "target": tier,
            "align": 8,
            "images": [{"name": "a.jpg", "width": width, "height": height}],
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["target"]["id"] == tier
    assert data["target"]["align"] == 8
    image = data["images"][0]
    assert set(image) == {
        "index", "name", "origin", "output", "scale", "mode", "aligned", "note"
    }
    assert image["index"] == 0
    assert image["name"] == "a.jpg"
    assert image["origin"] == {"width": width, "height": height}
    assert (image["output"]["width"], image["output"]["height"]) == expected
    assert image["mode"] == mode
    assert isinstance(image["note"], str) and image["note"]


def test_plan_aligned_flag_and_note(env):
    response = env.client.post(
        "/api/plan",
        json={"target": "1080p", "images": [{"name": "a.jpg", "width": 1000, "height": 667}]},
    )
    image = response.json()["images"][0]
    assert image["aligned"] is True
    assert "对齐" in image["note"]
    # 1280x720 -> 1920x1080 needs no snapping
    response = env.client.post(
        "/api/plan",
        json={"target": "1080p", "images": [{"name": "b.jpg", "width": 1280, "height": 720}]},
    )
    assert response.json()["images"][0]["aligned"] is False


def test_plan_multiple_images_and_align(env):
    response = env.client.post(
        "/api/plan",
        json={
            "target": "720p",
            "align": 16,
            "images": [
                {"name": "a.jpg", "width": 1280, "height": 720},
                {"name": "b.png", "width": 1080, "height": 1920},
            ],
        },
    )
    assert response.status_code == 200
    images = response.json()["images"]
    assert [item["index"] for item in images] == [0, 1]
    assert [item["name"] for item in images] == ["a.jpg", "b.png"]
    assert images[0]["output"] == {"width": 1280, "height": 720}
    assert images[1]["output"] == {"width": 720, "height": 1280}


@pytest.mark.parametrize(
    "payload",
    [
        {"target": "999", "images": [{"width": 10, "height": 10}]},
        {"target": "1080p", "images": []},
        {"target": "1080p"},
        {"target": "1080p", "images": [{"width": 0, "height": 10}]},
        {"target": "1080p", "images": [{"width": 10, "height": -1}]},
        {"target": "1080p", "images": [{"width": "x", "height": 10}]},
        {"target": "1080p", "images": "nope"},
        {"target": "1080p", "align": 1, "images": [{"width": 100, "height": 100}]},
    ],
)
def test_plan_invalid_payloads_are_400(env, payload):
    response = env.client.post("/api/plan", json=payload)
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, str) and detail


def test_plan_invalid_json_body_is_400_chinese(env):
    response = env.client.post(
        "/api/plan", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert "请求参数格式错误" in response.json()["detail"]


def test_plan_undecodable_http_body(env):
    response = env.client.post("/api/plan", json=[1, 2, 3])
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# /api/jobs lifecycle
# ---------------------------------------------------------------------------


def test_job_lifecycle(env):
    client = env.client
    png = image_bytes("PNG", (320, 240))
    jpg = image_bytes("JPEG", (200, 100))
    response = submit(client, [("a.png", png), ("b.jpg", jpg)], target="720p")
    assert response.status_code == 202, response.text
    created = response.json()
    assert len(created["job_id"]) == 12
    int(created["job_id"], 16)  # hex
    assert created["status"] in ("queued", "running")
    assert created["total"] == 2
    assert created["queue_position"] == 0
    assert created["target"] == {
        "id": "720p", "label": "720P", "rule": "short", "value": 720, "align": 8
    }
    assert created["engine"] == env.engine_info
    assert created["model"] == {
        "id": "auto", "label": "智能推荐", "denoise": False, "denoise_level": 0,
    }
    assert created["options"]["model"] == "auto"
    assert created["options"]["denoise"] is False
    assert created["options"]["denoise_level"] == 0
    assert created["files"][0]["model"] == "auto"
    assert created["files"][0]["denoise_level"] == 0
    assert "model_used" in created["files"][0]
    assert created["files"][0]["original"] == {"width": 320, "height": 240}
    assert created["files"][0]["output"] == {"width": 960, "height": 720}
    assert created["files"][0]["preview_url"] == f"/api/jobs/{created['job_id']}/files/0"
    assert created["files"][0]["download_url"].endswith("/files/0/download")
    job_id = created["job_id"]

    finished = wait_finished(client, job_id)
    assert finished["status"] == "done", finished
    assert finished["progress"] == 100.0
    assert finished["error"] is None
    assert finished["current_index"] is None
    assert finished["elapsed_sec"] >= 0.0
    assert finished["files"][0]["output_format"] == "png"  # auto: PNG input
    assert finished["files"][1]["output_format"] == "jpg"  # auto: JPEG input
    assert finished["files"][0]["filename"] == "a_720p.png"
    assert finished["files"][1]["filename"] == "b_720p.jpg"
    # auto resolves per file; both inputs are upscaled >2x -> x4plus
    assert finished["files"][0]["model_used"] == "RealESRGAN_x4plus"
    assert finished["files"][1]["model_used"] == "RealESRGAN_x4plus"
    for call in env.engine.calls:
        assert call["model"] == "auto"
        assert call["denoise"] is False
        assert call["denoise_level"] == 0
    for item in finished["files"]:
        assert item["status"] == "done"
        assert item["progress"] == 100.0
        assert item["output_size"] > 0
        expected = plan_target(
            item["original"]["width"], item["original"]["height"], "720p", align=8
        )["output"]
        assert item["output"] == expected

    # inline preview
    preview = client.get(f"/api/jobs/{job_id}/files/0")
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"
    assert preview.headers["cache-control"] == "no-cache"
    image = Image.open(io.BytesIO(preview.content))
    assert image.size == (960, 720)

    # attachment download with the generated file name
    download = client.get(f"/api/jobs/{job_id}/files/0/download")
    assert download.status_code == 200
    assert download.headers["content-type"] == "image/png"
    disposition = download.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert 'filename="a_720p.png"' in disposition
    assert len(download.content) == finished["files"][0]["output_size"]

    # custom filename query (ASCII + extension is enforced)
    custom = client.get(f"/api/jobs/{job_id}/files/1/download", params={"filename": "结果"})
    assert custom.status_code == 200
    header = custom.headers["content-disposition"]
    assert 'filename="download.jpg"' in header, header
    assert "filename*=UTF-8''" in header
    assert "%E7%BB%93%E6%9E%9C.jpg" in header

    # zip archive
    archive = client.get(f"/api/jobs/{job_id}/download.zip")
    assert archive.status_code == 200
    assert archive.headers["content-type"] == "application/zip"
    assert (
        archive.headers["content-disposition"]
        == f'attachment; filename="PicPolish_{job_id[:8]}.zip"'
    )
    with zipfile.ZipFile(io.BytesIO(archive.content)) as zf:
        names = sorted(zf.namelist())
        assert names == ["a_720p.png", "b_720p.jpg"]
        sizes = {info.filename: info.file_size for info in zf.infolist()}
        assert sizes["a_720p.png"] == finished["files"][0]["output_size"]

    # cancel after completion is a conflict
    conflict = client.post(f"/api/jobs/{job_id}/cancel")
    assert conflict.status_code == 409
    assert "无法取消" in conflict.json()["detail"]

    # delete removes the task and the cache folders
    deleted = client.delete(f"/api/jobs/{job_id}")
    assert deleted.status_code == 200
    payload = deleted.json()
    assert payload["deleted"] is True and payload["job_id"] == job_id
    assert payload["freed_bytes"] > 0
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert not (env.cache_root / "uploads" / job_id).exists()
    assert not (env.cache_root / "results" / job_id).exists()


def test_explicit_formats_and_quality(env):
    client = env.client
    noise = noise_image_bytes((320, 240), "PNG")

    def run(format_value: str, quality: int) -> dict:
        response = submit(
            client,
            [("noise.png", noise)],
            target="720p",
            format=format_value,
            quality=quality,
        )
        assert response.status_code == 202, response.text
        return wait_finished(client, response.json()["job_id"])

    png_job = run("png", 95)
    assert png_job["files"][0]["output_format"] == "png"
    png_download = client.get(f"/api/jobs/{png_job['job_id']}/files/0/download")
    assert png_download.headers["content-type"] == "image/png"

    jpg_low = run("jpg", 70)
    jpg_high = run("jpg", 100)
    assert jpg_low["files"][0]["output_format"] == "jpg"
    assert jpg_high["files"][0]["output_format"] == "jpg"
    jpg_download = client.get(f"/api/jobs/{jpg_high['job_id']}/files/0/download")
    assert jpg_download.headers["content-type"] == "image/jpeg"
    assert jpg_download.headers["content-disposition"].endswith('.jpg"')
    # quality must actually influence the encoded size
    assert jpg_low["files"][0]["output_size"] < jpg_high["files"][0]["output_size"]

    webp_job = run("webp", 80)
    assert webp_job["files"][0]["output_format"] == "webp"
    webp_download = client.get(f"/api/jobs/{webp_job['job_id']}/files/0/download")
    assert webp_download.headers["content-type"] == "image/webp"
    assert Image.open(io.BytesIO(webp_download.content)).size == (960, 720)


def test_alpha_png_stays_png_and_rgba_upload(env):
    rgba = image_bytes("PNG", (200, 120), (10, 20, 30), mode="RGBA")
    response = submit(env.client, [("alpha.png", rgba)], target="720p")
    finished = wait_finished(env.client, response.json()["job_id"])
    assert finished["status"] == "done"
    assert finished["files"][0]["output_format"] == "png"
    result = env.client.get(f"/api/jobs/{finished['job_id']}/files/0")
    assert Image.open(io.BytesIO(result.content)).mode in ("RGBA", "RGB")

    # A non-PNG (JPEG) input with format=auto becomes JPEG.
    jpg = image_bytes("JPEG", (200, 120))
    response = submit(env.client, [("plain.jpg", jpg)], target="720p")
    finished = wait_finished(env.client, response.json()["job_id"])
    assert finished["files"][0]["output_format"] == "jpg"


def test_duplicate_names_get_suffixes(env):
    client = env.client
    png = image_bytes("PNG", (160, 120))
    response = submit(client, [("same.png", png), ("same.png", png)], target="720p")
    finished = wait_finished(client, response.json()["job_id"])
    names = [item["filename"] for item in finished["files"]]
    assert names == ["same_720p.png", "same_720p_2.png"]
    archive = client.get(f"/api/jobs/{finished['job_id']}/download.zip")
    with zipfile.ZipFile(io.BytesIO(archive.content)) as zf:
        assert sorted(zf.namelist()) == ["same_720p.png", "same_720p_2.png"]


def test_upload_name_is_sanitized(env):
    client = env.client
    png = image_bytes("PNG", (160, 120))
    response = submit(client, [("../../evil name.png", png)], target="720p")
    created = response.json()
    assert created["files"][0]["name"] == "evil_name.png"
    finished = wait_finished(client, created["job_id"])
    assert finished["files"][0]["filename"] == "evil_name_720p.png"


def test_denoise_note_when_unavailable(env):
    png = image_bytes("PNG", (160, 120))
    response = submit(env.client, [("a.png", png)], target="720p", denoise="true")
    finished = wait_finished(env.client, response.json()["job_id"])
    assert finished["files"][0]["status"] == "done"
    assert "降噪" in (finished["files"][0]["note"] or "")


def test_wrong_engine_size_is_corrected(env):
    env.monkeypatch.setattr(jobs, "get_engine", lambda force=None: FakeEngine(wrong_size=True))
    png = image_bytes("PNG", (320, 240))
    response = submit(env.client, [("a.png", png)], target="720p")
    finished = wait_finished(env.client, response.json()["job_id"])
    assert finished["status"] == "done"
    expected = plan_target(320, 240, "720p", align=8)["output"]
    assert finished["files"][0]["output"] == expected
    result = env.client.get(f"/api/jobs/{finished['job_id']}/files/0")
    assert Image.open(io.BytesIO(result.content)).size == (
        expected["width"],
        expected["height"],
    )


def test_engine_failure_marks_job_error(env):
    env.monkeypatch.setattr(jobs, "get_engine", lambda force=None: FakeEngine(fail=True))
    response = submit(env.client, [("a.png", image_bytes())], target="720p")
    finished = wait_finished(env.client, response.json()["job_id"])
    assert finished["status"] == "error"
    assert "处理失败" in finished["error"]
    assert finished["files"][0]["status"] == "error"
    assert "fake engine boom" in finished["files"][0]["error"]


def test_engine_init_failure_marks_job_error(env):
    def broken_engine(force=None):
        raise RuntimeError("没有可用的推理后端")

    env.monkeypatch.setattr(jobs, "get_engine", broken_engine)
    response = submit(env.client, [("a.png", image_bytes())], target="720p")
    finished = wait_finished(env.client, response.json()["job_id"])
    assert finished["status"] == "error"
    assert "超分引擎初始化失败" in finished["error"]
    assert finished["files"][0]["status"] == "error"


# ---------------------------------------------------------------------------
# model selection / denoise parameters (docs/API.md §6, §7, §14)
# ---------------------------------------------------------------------------


def test_job_model_auto_resolves_per_file(env):
    client = env.client
    # 500x400 @720p -> ratio 1.8 -> x2plus; 320x240 @720p -> ratio 3 -> x4plus
    wide = image_bytes("PNG", (500, 400))
    small = image_bytes("PNG", (320, 240))
    created, finished = submit_and_wait(
        client, [("wide.png", wide), ("small.png", small)], target="720p"
    )
    assert created["model"]["id"] == "auto"
    assert finished["status"] == "done", finished
    used = [item["model_used"] for item in finished["files"]]
    assert used == ["RealESRGAN_x2plus", "RealESRGAN_x4plus"], used
    assert [call["model"] for call in env.engine.calls] == ["auto", "auto"]
    ratios = sorted(round(call[1], 6) for call in env.engine.resolve_calls)
    assert ratios == [1.8, 3.0], env.engine.resolve_calls


def test_job_model_anime6b(env):
    created, finished = submit_and_wait(
        env.client, [("a.png", image_bytes())], target="720p", model="realesrgan-anime6b"
    )
    assert created["model"] == {
        "id": "realesrgan-anime6b",
        "label": "Real-ESRGAN 动漫",
        "denoise": False,
        "denoise_level": 0,
    }
    assert finished["status"] == "done", finished
    item = finished["files"][0]
    assert item["model"] == "realesrgan-anime6b"
    assert item["model_used"] == "RealESRGAN_x4plus_anime_6B"
    assert env.engine.calls[-1]["model"] == "realesrgan-anime6b"


def test_job_realcugan_denoise_levels(env):
    client = env.client
    # denoise=true without level -> level 2 (API.md §14.3); ratio 3 -> native 3x
    created, finished = submit_and_wait(
        client, [("a.png", image_bytes())], target="720p",
        model="realcugan", denoise="true",
    )
    assert created["model"] == {
        "id": "realcugan",
        "label": "Real-CUGAN（动漫/视频）",
        "denoise": True,
        "denoise_level": 2,
    }
    assert created["files"][0]["denoise_level"] == 2
    assert finished["status"] == "done", finished
    item = finished["files"][0]
    assert item["model"] == "realcugan"
    assert item["model_used"] == "Real-CUGAN 3x"
    assert item["denoise_level"] == 3  # 3x/4x mapping: denoise3x
    call = env.engine.calls[-1]
    assert call["model"] == "realcugan"
    assert call["denoise"] is True
    assert call["denoise_level"] == 2

    # explicit level is forwarded as-is
    _created2, finished2 = submit_and_wait(
        client, [("a.png", image_bytes())], target="720p",
        model="realcugan", denoise_level="3",
    )
    call = env.engine.calls[-1]
    assert call["denoise_level"] == 3
    assert finished2["files"][0]["denoise_level"] == 3


def test_job_realcugan_conservative_ignores_level(env):
    created, finished = submit_and_wait(
        env.client, [("a.png", image_bytes())], target="720p",
        model="realcugan-conservative", denoise="true", denoise_level="3",
    )
    assert created["model"]["id"] == "realcugan-conservative"
    assert created["model"]["denoise_level"] == 0
    assert finished["status"] == "done", finished
    item = finished["files"][0]
    assert item["model_used"] == "Real-CUGAN conservative"
    assert item["denoise_level"] == 0
    assert "保守" in (item["note"] or "")


def test_job_rrdb_denoise_level_normalized(env):
    created, finished = submit_and_wait(
        env.client, [("a.png", image_bytes())], target="720p",
        model="realesrgan-x4plus", denoise="true", denoise_level="3",
    )
    assert created["model"]["denoise"] is True
    assert created["model"]["denoise_level"] == 1  # rrdb only knows 0/1
    assert finished["status"] == "done", finished
    call = env.engine.calls[-1]
    assert call["model"] == "realesrgan-x4plus"
    assert call["denoise"] is True
    assert call["denoise_level"] == 1


def test_job_invalid_model_is_400(env):
    response = submit(env.client, [("a.png", image_bytes())], model="foo")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "未知的 model" in detail
    assert "auto" in detail


def test_job_unavailable_model_is_400(env):
    env.monkeypatch.setattr(
        jobs, "list_models", lambda: make_models_payload(available={"auto"})
    )
    response = submit(env.client, [("a.png", image_bytes())], model="realcugan")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "权重缺失" in detail
    assert "fetch_models" in detail


@pytest.mark.parametrize("value", ["-1", "4", "abc"])
def test_job_invalid_denoise_level_is_400(env, value):
    response = submit(env.client, [("a.png", image_bytes())], denoise_level=value)
    assert response.status_code == 400
    assert "denoise_level" in response.json()["detail"]


def test_pil_downgrade_note(env):
    env.monkeypatch.setattr(
        jobs, "get_engine", lambda force=None: FakeEngine(backend="pil")
    )
    created, finished = submit_and_wait(
        env.client, [("a.png", image_bytes())], target="720p",
        model="realesrgan-x4plus",
    )
    assert finished["status"] == "done", finished
    item = finished["files"][0]
    assert item["model_used"] == "RealESRGAN_x4plus"
    note = item["note"] or ""
    assert "降级" in note and "兼容模式" in note, note


def test_real_model_registry_contract():
    """The real ``app/models.py`` registry must match the API.md §14.1 shape."""
    try:
        from app import models as models_module
    except ImportError:  # pragma: no cover - registry ships with the project
        pytest.skip("app.models is not available")

    payload = models_module.list_models()
    assert payload["default"] == "auto"
    ids = [model["id"] for model in payload["models"]]
    assert ids == MODEL_IDS[: len(ids)]
    assert len(ids) == len(MODEL_IDS)
    required = {
        "id", "label", "description", "tags", "recommended_for", "family",
        "native_scales", "denoise", "license", "size_mb", "source", "available",
    }
    for model in payload["models"]:
        assert required <= set(model)
        assert model["family"] in ("rrdb", "cugan")
        assert {"supported", "native", "levels", "default"} <= set(model["denoise"])
        assert model["denoise"]["levels"]
        assert isinstance(model["available"], bool)
    available = models_module.available_model_ids()
    assert available == [model["id"] for model in payload["models"] if model["available"]]
    assert payload["models"][0]["id"] == "auto"
    assert payload["models"][-1]["id"] == "realcugan-conservative"


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


def test_cancel_queued_job(env):
    client = env.client
    gate = threading.Event()
    slow = FakeEngine(gate=gate)
    env.monkeypatch.setattr(jobs, "get_engine", lambda force=None: slow)

    first = submit(client, [("first.png", image_bytes())], target="720p").json()
    wait_status(client, first["job_id"], "running")

    second_response = submit(client, [("second.png", image_bytes())], target="720p")
    assert second_response.status_code == 202
    second = second_response.json()
    assert second["queue_position"] == 1

    started = time.monotonic()
    cancelled = client.post(f"/api/jobs/{second['job_id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json() == {"job_id": second["job_id"], "status": "cancelled"}
    assert time.monotonic() - started < 3.0  # queued cancellation is immediate
    queued_state = _get_job(client, second["job_id"])
    assert queued_state["status"] == "cancelled"
    assert queued_state["files"][0]["status"] == "cancelled"

    again = client.post(f"/api/jobs/{second['job_id']}/cancel")
    assert again.status_code == 409

    gate.set()
    assert wait_finished(client, first["job_id"])["status"] == "done"
    # the cancelled job stays cancelled and is never processed
    assert _get_job(client, second["job_id"])["status"] == "cancelled"


def test_cancel_running_job(env):
    client = env.client
    slow = FakeEngine(steps=20, loop_delay=0.05)
    env.monkeypatch.setattr(jobs, "get_engine", lambda force=None: slow)
    created = submit(client, [("a.png", image_bytes())], target="720p").json()
    job_id = created["job_id"]

    def progressed():
        data = _get_job(client, job_id)
        return data if data["files"][0]["progress"] > 0 else None

    _wait_for(progressed)
    response = client.post(f"/api/jobs/{job_id}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    item = wait_file_status(client, job_id, 0, "cancelled")
    assert item["status"] == "cancelled"
    final = _get_job(client, job_id)
    assert final["status"] == "cancelled"
    assert final["finished_at"] is not None
    assert client.get(f"/api/jobs/{job_id}/files/0").status_code == 409


def test_results_not_ready_conflicts(env):
    client = env.client
    gate = threading.Event()
    gated = FakeEngine(gate=gate)
    env.monkeypatch.setattr(jobs, "get_engine", lambda force=None: gated)
    created = submit(client, [("a.png", image_bytes())], target="720p").json()
    job_id = created["job_id"]
    wait_status(client, job_id, "running")
    assert client.get(f"/api/jobs/{job_id}/files/0").status_code == 409
    assert client.get(f"/api/jobs/{job_id}/files/0/download").status_code == 409
    assert client.get(f"/api/jobs/{job_id}/download.zip").status_code == 409
    try:
        detail = client.get(f"/api/jobs/{job_id}/files/0").json()["detail"]
        assert "结果尚未生成" in detail
    finally:
        gate.set()
    assert wait_finished(client, job_id)["status"] == "done"


# ---------------------------------------------------------------------------
# validation / error mapping
# ---------------------------------------------------------------------------


def test_submit_missing_target(env):
    response = env.client.post(
        "/api/jobs", files=[("files", ("a.png", image_bytes(), "image/png"))]
    )
    assert response.status_code == 400
    assert "target" in response.json()["detail"]


def test_submit_without_files(env):
    response = env.client.post("/api/jobs", data={"target": "720p"})
    assert response.status_code == 400
    assert "至少上传" in response.json()["detail"]


def test_submit_unsupported_type(env):
    response = submit(env.client, [("notes.txt", b"hello world")], target="720p")
    assert response.status_code == 415
    assert "不支持的文件类型" in response.json()["detail"]


def test_submit_unknown_target(env):
    response = submit(env.client, [("a.png", image_bytes())], target="8k")
    assert response.status_code == 400
    assert "未知的 target" in response.json()["detail"]


def test_submit_corrupt_image(env):
    response = submit(env.client, [("bad.png", b"definitely not an image")], target="720p")
    assert response.status_code == 400
    assert "无法解析图片" in response.json()["detail"]


def test_submit_empty_file(env):
    response = submit(env.client, [("empty.png", b"")], target="720p")
    assert response.status_code == 400
    assert "空文件" in response.json()["detail"]


def test_submit_file_too_large(env):
    env.monkeypatch.setattr(config, "MAX_FILE_MB", 1)
    response = submit(
        env.client, [("big.png", image_bytes() + b"\x00" * (2 * 1024 * 1024))], target="720p"
    )
    assert response.status_code == 413
    assert "MB" in response.json()["detail"]


def test_submit_too_many_files(env):
    env.monkeypatch.setattr(config, "MAX_FILES", 1)
    png = image_bytes("PNG", (64, 64))
    response = submit(env.client, [("a.png", png), ("b.png", png)], target="720p")
    assert response.status_code == 400
    assert "最多" in response.json()["detail"]


def test_submit_invalid_options(env):
    png = image_bytes()
    assert submit(env.client, [("a.png", png)], format="gif").status_code == 400
    assert submit(env.client, [("a.png", png)], quality="10").status_code == 400
    assert submit(env.client, [("a.png", png)], align="1").status_code == 400
    assert submit(env.client, [("a.png", png)], tile="32").status_code == 400
    assert submit(env.client, [("a.png", png)], denoise="maybe").status_code == 400


def test_unknown_job_endpoints_are_404(env):
    client = env.client
    assert client.get("/api/jobs/deadbeef0000").status_code == 404
    assert client.get("/api/jobs/deadbeef0000/files/0").status_code == 404
    assert client.get("/api/jobs/deadbeef0000/files/0/download").status_code == 404
    assert client.get("/api/jobs/deadbeef0000/download.zip").status_code == 404
    assert client.post("/api/jobs/deadbeef0000/cancel").status_code == 404
    assert client.delete("/api/jobs/deadbeef0000").status_code == 404
    assert "任务不存在" in client.get("/api/jobs/deadbeef0000").json()["detail"]


def test_file_index_out_of_range(env):
    created = submit(env.client, [("a.png", image_bytes())], target="720p").json()
    job_id = created["job_id"]
    wait_finished(env.client, job_id)
    assert env.client.get(f"/api/jobs/{job_id}/files/5").status_code == 404
    assert env.client.get(f"/api/jobs/{job_id}/files/-1").status_code == 404


def test_delete_running_job_conflicts(env):
    gate = threading.Event()
    env.monkeypatch.setattr(jobs, "get_engine", lambda force=None: FakeEngine(gate=gate))
    created = submit(env.client, [("a.png", image_bytes())], target="720p").json()
    job_id = created["job_id"]
    wait_status(env.client, job_id, "running")
    try:
        response = env.client.delete(f"/api/jobs/{job_id}")
        assert response.status_code == 409
        assert "处理中" in response.json()["detail"]
    finally:
        gate.set()
    assert wait_finished(env.client, job_id)["status"] == "done"


# ---------------------------------------------------------------------------
# TTL cleanup / shutdown
# ---------------------------------------------------------------------------


def test_ttl_cleanup_removes_finished_jobs(env):
    created = submit(env.client, [("a.png", image_bytes())], target="720p").json()
    job_id = created["job_id"]
    wait_finished(env.client, job_id)
    env.manager._keep_hours = 0.0  # noqa: SLF001 - direct TTL unit test
    stats = env.manager.cleanup_expired()
    assert stats["removed_jobs"] == 1
    assert stats["freed_bytes"] > 0
    assert env.client.get(f"/api/jobs/{job_id}").status_code == 404
    assert not (env.cache_root / "results" / job_id).exists()

    # orphan folders (left over after a crash) are cleaned as well
    orphan = env.cache_root / "uploads" / "0123456789ab"
    orphan.mkdir(parents=True)
    (orphan / "x.bin").write_bytes(b"x" * 10)
    stats = env.manager.cleanup_expired()
    assert stats["freed_bytes"] >= 10
    assert not orphan.exists()


def test_shutdown_clears_cache_when_clean_on_exit(env):
    env.monkeypatch.setattr(config, "CLEAN_ON_EXIT", True)
    with TestClient(main_module.app) as client:
        created = submit(client, [("a.png", image_bytes())], target="720p").json()
        assert wait_finished(client, created["job_id"])["status"] == "done"
        assert any((env.cache_root / "results").rglob("*"))
    for folder_name in ("uploads", "results", "tmp"):
        folder = env.cache_root / folder_name
        assert folder.is_dir()
        assert [item for item in folder.rglob("*") if item.is_file()] == []
    assert env.client.get("/api/health").status_code == 200


# ---------------------------------------------------------------------------
# static frontend + low level helpers
# ---------------------------------------------------------------------------


def test_index_and_static(env):
    index = env.client.get("/")
    assert index.status_code == 200
    assert "text/html" in index.headers["content-type"]
    static = env.client.get("/web/index.html")
    assert static.status_code == 200
    assert "text/html" in static.headers["content-type"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../evil name.png", "evil_name.png"),
        ("C:\\temp\\a.txt", "a.txt"),
        ("CON.png", "_CON.png"),
        ("", "image"),
        ("...", "image"),
        ("正常名字.jpg", "正常名字.jpg"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert jobs.sanitize_filename(raw) == expected


def test_sanitize_filename_truncates():
    cleaned = jobs.sanitize_filename("a" * 300 + ".png")
    assert len(cleaned) <= 120
    assert cleaned.endswith(".png")


def test_content_disposition_helpers():
    assert jobs.content_disposition("a_720p.jpg") == 'attachment; filename="a_720p.jpg"'
    # pure Chinese file name: ascii fallback must keep a usable stem + extension
    header = jobs.content_disposition("我的照片.jpg")
    assert header.startswith("attachment;")
    assert 'filename="download.jpg"' in header, header
    assert "filename*=UTF-8''" in header
    assert "%E6%88%91%E7%9A%84%E7%85%A7%E7%89%87.jpg" in header
    # mixed name keeps the ascii part
    mixed = jobs.content_disposition("结果_720p.jpg")
    assert 'filename="_720p.jpg"' in mixed
    assert "filename*=UTF-8''" in mixed
    assert jobs.sanitize_download_filename(None, "jpg") is None
    assert jobs.sanitize_download_filename("我的照片", "webp") == "我的照片.webp"
    assert jobs.sanitize_download_filename("../../x", "png") == "x.png"
