#!/usr/bin/env python
"""PicPolish Windows full-chain E2E test (API level).

The server lifecycle (start/stop + cache cleanup assertions) is owned by
``tests/e2e/e2e_windows.ps1``; this script assumes a PicPolish instance is
already listening at ``--base-url`` and never starts or kills a process.

What it verifies, strictly against ``docs/API.md``:

1. ``GET /api/health`` -- engine/limits/cache block shape + expected engine.
2. ``GET /api/targets`` -- exactly the four documented tiers.
3. ``POST /api/plan`` -- the exact output-size table from docs/API.md 1.1
   (including the 8 px snapping rule and the downscale case).
4. ``POST /api/jobs`` -- a 4-file multipart job (jpg / png / portrait / RGBA)
   -> poll -> every file's downloaded bytes re-opened with PIL and compared to
   the plan (exact size + format), preview/download headers, Content-Disposition
   filename, ZIP entries and bytes, DELETE + freed_bytes.
5. Negative paths: 404 / 400 / 415 / 409 per docs/API.md 13.
6. Server-local cache layout (cache/uploads|results/<job_id>) when run on the
   same machine.

Exit code: ``0`` when every blocking check passes, ``1`` when any FAILs,
``2`` when the server is unreachable / the harness itself crashed.

Usage::

    .venv\\Scripts\\python.exe tests\\e2e\\e2e_windows.py \\
        --base-url http://127.0.0.1:8796 --expect-engine realesrgan --expect-device cuda
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tempfile
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CACHE_DIR,
    MODEL_IDS,
    MODELS_EXPECTED,
    PROJECT_ROOT,
    SAMPLES_DIR,
    TIERS_EXPECTED,
    ApiClient,
    ApiError,
    Checker,
    approx,
    bicubic_baseline,
    cache_contents,
    content_disposition_filename,
    content_problems,
    image_info,
    laplacian_variance,
    load_samples,
    model_used_matches,
    parse_content_disposition,
    setup_console,
    tile_metrics,
    write_json,
)

TIER = "1080p"
ALIGN = 8

# --- docs/API.md 1.1 exact table (+ a few extra contract corner cases) -----
PLAN_CASES: list[dict] = [
    # w, h, tier, align, out_w, out_h, scale, mode, aligned
    dict(w=1280, h=720, tier="1080p", out=(1920, 1080), scale=1.5, mode="upscale",
         aligned=False, why="API.md 1.1 row 1"),
    dict(w=1920, h=1080, tier="2k", out=(2560, 1440), scale=2560 / 1920, mode="upscale",
         aligned=False, why="API.md 1.1 row 2"),
    dict(w=1000, h=667, tier="1080p", out=(1616, 1080), scale=1080 / 667, mode="upscale",
         aligned=True, why="API.md 1.1 row 3 (snap 1619 -> 1616)"),
    dict(w=1080, h=1920, tier="720p", out=(720, 1280), scale=720 / 1080, mode="downscale",
         aligned=False, why="API.md 1.1 row 4 (portrait)"),
    dict(w=3840, h=2160, tier="1080p", out=(1920, 1080), scale=0.5, mode="downscale",
         aligned=False, why="API.md 1.1 row 5 (downscale)"),
    # extra corners derived from the same rule
    dict(w=720, h=1280, tier="1080p", out=(1080, 1920), scale=1.5, mode="upscale",
         aligned=False, why="portrait upscale"),
    dict(w=320, h=240, tier="1080p", out=(1440, 1080), scale=4.5, mode="upscale",
         aligned=False, why="tiny upscale"),
    dict(w=800, h=600, tier="1080p", out=(1440, 1080), scale=1.8, mode="upscale",
         aligned=False, why="RGBA sample size"),
    dict(w=1920, h=1200, tier="1080p", out=(1728, 1080), scale=0.9, mode="downscale",
         aligned=False, why="real photo downscale"),
    dict(w=3840, h=1200, tier="1080p", out=(3456, 1080), scale=0.9, mode="downscale",
         aligned=False, why="4K panorama downscale"),
    dict(w=1280, h=720, tier="720p", out=(1280, 720), scale=1.0, mode="same",
         aligned=False, why="same size"),
    dict(w=1000, h=667, tier="1080p", align=4, out=(1620, 1080), scale=1080 / 667,
         mode="upscale", aligned=True, why="align=4 snapping -> 1620"),
    dict(w=700, h=700, tier="1080p", out=(1080, 1080), scale=1080 / 700, mode="upscale",
         aligned=False, why="square gradient tile probe"),
]

# Tile-seam probe: a near-uniform gentle gradient must stay uniform and
# seam-free after multi-tile inference (a broken tiler leaks black blocks).
GRADIENT_SAMPLE = "solid_gradient_700x700.png"
GRADIENT_OUT = (1080, 1080)
TILE_GRID = 4
TILE_MEAN_TOL = 5.0      # Lead-mandated: per-tile mean vs global mean < 5
SEAM_TOL = 3.0           # max jump between adjacent column/row means (levels)

# Real produced bytes per tier: every tier and every special geometry must come
# out of the pipeline with exactly the /api/plan pixel size (not plan-only).
TIER_OUTPUT_CASES = [
    # sample, tier, expected (w, h), pil format
    ("photo_1280x720.jpg", "720p", (1280, 720), "JPEG"),      # same-size path
    ("photo_1280x720.jpg", "2k", (2560, 1440), "JPEG"),       # x2 model native
    ("photo_1280x720.jpg", "4k", (3840, 2160), "JPEG"),       # x4 model upscale
    ("odd_1000x667.jpg", "1080p", (1616, 1080), "JPEG"),      # 8px snapped output
    ("portrait_720x1280.jpg", "720p", (720, 1280), "JPEG"),   # portrait downscale
    ("large_3840x2160.jpg", "1080p", (1920, 1080), "JPEG"),   # 4K input downscale
]

# Main job: 4 files covering jpg / png / portrait / RGBA (format=auto).
# name -> (out_w, out_h, api_format, pil_format, mime, stem)
JOB_FILES = [
    ("photo_1280x720.jpg", (1920, 1080), "jpg", "JPEG", "image/jpeg"),
    ("text_gradient_1280x720.png", (1920, 1080), "png", "PNG", "image/png"),
    ("portrait_720x1280.jpg", (1080, 1920), "jpg", "JPEG", "image/jpeg"),
    ("alpha_800x600.png", (1440, 1080), "png", "PNG", "image/png"),
]

TERMINAL = {"done", "error", "cancelled"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def expected_result_name(original: str, tier: str, ext: str) -> str:
    return f"{Path(original).stem}_{tier}.{ext}"


def guarded(chk: Checker, category: str, fn) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - one broken section must not kill the run
        chk.fail(f"{category}.exception", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Section 1: health + targets
# ---------------------------------------------------------------------------
def section_health(chk: Checker, client: ApiClient, args) -> dict:
    resp = client.get("/api/health")
    if not chk.expect_eq("health.http_status", resp.status_code, 200, "health"):
        return {}
    health = resp.json()

    chk.expect_eq("health.status", health.get("status"), "ok", "health")
    chk.expect_eq("health.app", health.get("app"), "PicPolish", "health")
    chk.expect("health.version", isinstance(health.get("version"), str) and bool(health["version"]),
               f"version={health.get('version')!r}", "health")

    engine = health.get("engine")
    if not chk.expect("health.engine_is_object", isinstance(engine, dict), f"engine={engine!r}",
                      "health"):
        return health
    for key in ("backend", "device", "gpu", "model", "native_scale"):
        chk.expect(f"health.engine.has_{key}", key in engine, f"engine keys={sorted(engine)}",
                   "health")
    backend = engine.get("backend")
    device = engine.get("device")
    chk.expect_in("health.engine.backend", backend, ("realesrgan", "pil"), "health")
    chk.expect_in("health.engine.device", device, ("cuda", "cpu"), "health")
    native = engine.get("native_scale")
    chk.expect("health.engine.native_scale_int", isinstance(native, int) and native > 0,
               f"native_scale={native!r}", "health")
    if backend == "realesrgan":
        chk.expect("health.engine.model_set",
                   isinstance(engine.get("model"), str) and bool(engine.get("model")),
                   f"model={engine.get('model')!r}", "health")
        chk.expect_in("health.engine.native_scale_x", native, (2, 4), "health")
    if args.expect_engine:
        chk.expect_eq("health.engine.expected_backend", backend, args.expect_engine, "health")
    if args.expect_device:
        chk.expect_eq("health.engine.expected_device", device, args.expect_device, "health")
    print(f"      engine: backend={backend!r} device={device!r} gpu={engine.get('gpu')!r} "
          f"model={engine.get('model')!r} native_scale={native!r}", flush=True)

    limits = health.get("limits")
    if chk.expect("health.limits_is_object", isinstance(limits, dict), f"limits={limits!r}",
                  "health"):
        for key in ("max_files", "max_file_mb", "max_total_mb", "max_pixels"):
            chk.expect(f"health.limits.has_{key}",
                       isinstance(limits.get(key), int) and limits[key] > 0,
                       f"limits[{key}]={limits.get(key)!r}", "health")
        for key, default in (("max_files", 20), ("max_file_mb", 50),
                             ("max_total_mb", 200), ("max_pixels", 20_000_000)):
            if key in limits and limits[key] != default:
                chk.warn(f"health.limits.{key}_non_default",
                         f"{limits[key]} != default {default} (env override?)", "health")

    cache = health.get("cache")
    if chk.expect("health.cache_is_object", isinstance(cache, dict), f"cache={cache!r}",
                  "health"):
        chk.expect("health.cache.has_jobs", isinstance(cache.get("jobs"), int),
                   f"cache={cache!r}", "health")
        chk.expect("health.cache.has_result_mb",
                   isinstance(cache.get("result_mb"), (int, float)), f"cache={cache!r}", "health")
        if args.expect_fresh:
            chk.expect_eq("health.cache.fresh_jobs", cache.get("jobs"), 0, "health")

    # v1.1: health carries the model summary (docs/API.md section 3 / 14).
    models = health.get("models")
    if chk.expect("health.models_is_object", isinstance(models, dict), f"models={models!r}",
                  "health"):
        chk.expect_eq("health.models.default", models.get("default"), "auto", "health")
        available = models.get("available")
        chk.expect("health.models.available_is_list", isinstance(available, list),
                   f"available={available!r}", "health")
        if isinstance(available, list):
            chk.expect_eq("health.models.available_ids", sorted(available), sorted(MODEL_IDS),
                          "health")
        print(f"      models: default={models.get('default')!r} "
              f"available={models.get('available')!r}", flush=True)
    return health


def section_models_api(chk: Checker, client: ApiClient, artifact_dir: Path) -> dict:
    """``GET /api/models`` contract + job model/denoise parameter validation."""
    started = time.monotonic()
    resp = client.get("/api/models")
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if not chk.expect_eq("models.http_status", resp.status_code, 200, "models"):
        return {}
    elapsed_s = time.monotonic() - started
    payload = resp.json()
    chk.expect("models.fast_response", elapsed_s < 2.0,
               f"/api/models answered in {elapsed_ms:.0f} ms (must not load torch)", "models")
    chk.expect_eq("models.default", payload.get("default"), "auto", "models")
    models = payload.get("models")
    if not chk.expect("models.is_list", isinstance(models, list), f"models={models!r}",
                      "models"):
        return payload
    chk.expect_eq("models.count", len(models), len(MODEL_IDS), "models")
    chk.expect_eq("models.ids_order", [m.get("id") for m in models], MODEL_IDS, "models")

    for expected, actual in zip(MODELS_EXPECTED, models):
        cid = f"models.{expected['id']}"
        if not isinstance(actual, dict):
            chk.fail(f"{cid}.object", f"entry={actual!r}", "models")
            continue
        chk.expect(f"{cid}.label", isinstance(actual.get("label"), str) and bool(actual["label"]),
                   f"label={actual.get('label')!r}", "models")
        chk.expect(f"{cid}.description",
                   isinstance(actual.get("description"), str) and bool(actual["description"]),
                   "", "models")
        chk.expect(f"{cid}.family", actual.get("family") == expected["family"],
                   f"family={actual.get('family')!r} expected {expected['family']!r}", "models")
        scales = actual.get("native_scales")
        chk.expect(f"{cid}.native_scales",
                   isinstance(scales, list) and len(scales) > 0
                   and all(isinstance(s, int) and s > 0 for s in scales),
                   f"native_scales={scales!r}", "models")
        tags = actual.get("tags")
        chk.expect(f"{cid}.tags", isinstance(tags, list) and len(tags) > 0,
                   f"tags={tags!r}", "models")
        rec = actual.get("recommended_for")
        chk.expect(f"{cid}.recommended_for", isinstance(rec, list) and len(rec) > 0,
                   f"recommended_for={rec!r}", "models")
        denoise = actual.get("denoise")
        if chk.expect(f"{cid}.denoise_is_object", isinstance(denoise, dict),
                      f"denoise={denoise!r}", "models"):
            chk.expect(f"{cid}.denoise.supported",
                       isinstance(denoise.get("supported"), bool), "", "models")
            chk.expect(f"{cid}.denoise.native",
                       denoise.get("native") == expected["denoise_native"],
                       f"native={denoise.get('native')!r} expected {expected['denoise_native']!r}",
                       "models")
            levels = denoise.get("levels")
            chk.expect(f"{cid}.denoise.levels",
                       isinstance(levels, list) and len(levels) > 0
                       and all(isinstance(level, int) and 0 <= level <= 3 for level in levels),
                       f"levels={levels!r}", "models")
            chk.expect(f"{cid}.denoise.default",
                       isinstance(denoise.get("default"), int)
                       and (not isinstance(levels, list) or denoise["default"] in levels),
                       f"default={denoise.get('default')!r} levels={levels!r}", "models")
        if expected["id"] == "realcugan":
            chk.expect_eq(f"{cid}.denoise.levels_full", actual.get("denoise", {}).get("levels"),
                          [0, 1, 2, 3], "models")
        chk.expect(f"{cid}.license", isinstance(actual.get("license"), str)
                   and bool(actual["license"]), f"license={actual.get('license')!r}", "models")
        size_mb = actual.get("size_mb")
        chk.expect(f"{cid}.size_mb", isinstance(size_mb, (int, float)) and size_mb > 0,
                   f"size_mb={size_mb!r}", "models")
        chk.expect(f"{cid}.source", isinstance(actual.get("source"), str)
                   and bool(actual["source"]), f"source={actual.get('source')!r}", "models")
        chk.expect_eq(f"{cid}.available", actual.get("available"), True, "models")

    # -- job parameter validation & echo (docs/API.md 6/14.4) ------------
    tiny = SAMPLES_DIR / "tiny_320x240.png"
    bad_model = client.create_job([tiny], target="720p", model="not-a-model")
    chk.expect_eq("models.job.invalid_model_status", bad_model.status_code, 400, "models")
    chk.expect("models.job.invalid_model_detail",
               isinstance((bad_model.json() or {}).get("detail"), str)
               if bad_model.status_code == 400 else False,
               f"body={bad_model.text[:200]!r}", "models")
    bad_level = client.create_job([tiny], target="720p", model="realcugan", denoise_level=4)
    chk.expect_eq("models.job.invalid_denoise_level_status", bad_level.status_code, 400,
                  "models")

    # Create (and immediately cancel) one job to check the model echo fields.
    created_resp = client.create_job([tiny], target="720p", model="realesrgan-x2plus",
                                     denoise="false")
    if chk.expect_eq("models.job.create_status", created_resp.status_code, 202, "models"):
        created = created_resp.json()
        job_id = created.get("job_id")
        model_obj = created.get("model")
        if chk.expect("models.job.model_object", isinstance(model_obj, dict),
                      f"model={model_obj!r}", "models"):
            chk.expect_eq("models.job.model.id", model_obj.get("id"), "realesrgan-x2plus",
                          "models")
            chk.expect("models.job.model.label", isinstance(model_obj.get("label"), str)
                       and bool(model_obj["label"]), f"label={model_obj.get('label')!r}",
                       "models")
            chk.expect_eq("models.job.model.denoise", model_obj.get("denoise"), False,
                          "models")
            chk.expect("models.job.model.denoise_level",
                       isinstance(model_obj.get("denoise_level"), int),
                       f"denoise_level={model_obj.get('denoise_level')!r}", "models")
        files = created.get("files") or []
        if files:
            entry = files[0]
            chk.expect_eq("models.job.file.model", entry.get("model"), "realesrgan-x2plus",
                          "models")
            chk.expect("models.job.file.model_used_key", "model_used" in entry,
                       f"keys={sorted(entry)}", "models")
            chk.expect("models.job.file.denoise_level_key", "denoise_level" in entry,
                       f"keys={sorted(entry)}", "models")
        if isinstance(job_id, str) and job_id:
            client.cancel_job(job_id)
            try:
                final, _wall, _samples = client.wait_job(job_id, timeout=60)
                if final.get("status") == "done":
                    term_entry = (final.get("files") or [{}])[0]
                    used_ok, used_detail = model_used_matches(
                        "realesrgan-x2plus", term_entry.get("model_used"))
                    chk.expect("models.job.file.model_used_resolved", used_ok, used_detail,
                               "models")
            except Exception as exc:  # noqa: BLE001
                chk.warn("models.job.cancel_wait", f"{type(exc).__name__}: {exc}", "models")
            finally:
                try:
                    client.delete_job(job_id)
                except Exception:  # noqa: BLE001
                    pass
    return payload


def section_targets(chk: Checker, client: ApiClient) -> None:
    resp = client.get("/api/targets")
    if not chk.expect_eq("targets.http_status", resp.status_code, 200, "targets"):
        return
    targets = resp.json().get("targets")
    if not chk.expect("targets.is_list", isinstance(targets, list), f"targets={targets!r}",
                      "targets"):
        return
    chk.expect_eq("targets.count", len(targets), 4, "targets")
    for index, expected in enumerate(TIERS_EXPECTED):
        if index >= len(targets):
            chk.fail(f"targets[{index}].present", "missing tier", "targets")
            continue
        actual = targets[index]
        for key in ("id", "label", "rule", "value"):
            chk.expect_eq(f"targets[{index}].{key}", actual.get(key), expected[key], "targets")
        hint = actual.get("hint")
        chk.expect(f"targets[{index}].hint", isinstance(hint, str) and len(hint) > 0,
                   f"hint={hint!r}", "targets")


# ---------------------------------------------------------------------------
# Section 2: /api/plan exact output-size table
# ---------------------------------------------------------------------------
def section_plan(chk: Checker, client: ApiClient) -> None:
    for case in PLAN_CASES:
        align = case.get("align", ALIGN)
        cid = f"plan.{case['w']}x{case['h']}@{case['tier']}" + (
            f".align{align}" if align != ALIGN else "")
        name = f"sample_{case['w']}x{case['h']}.img"
        resp = client.plan([{"name": name, "width": case["w"], "height": case["h"]}],
                           case["tier"], align=align)
        if not chk.expect_eq(f"{cid}.http_status", resp.status_code, 200, "plan"):
            continue
        payload = resp.json()

        target = payload.get("target") or {}
        chk.expect_eq(f"{cid}.target.id", target.get("id"), case["tier"], "plan")
        chk.expect_eq(f"{cid}.target.align", target.get("align"), align, "plan")
        expected_tier = next(t for t in TIERS_EXPECTED if t["id"] == case["tier"])
        chk.expect_eq(f"{cid}.target.label", target.get("label"), expected_tier["label"], "plan")
        chk.expect_eq(f"{cid}.target.rule", target.get("rule"), expected_tier["rule"], "plan")
        chk.expect_eq(f"{cid}.target.value", target.get("value"), expected_tier["value"], "plan")

        images = payload.get("images")
        if not chk.expect(f"{cid}.images_list", isinstance(images, list) and len(images) == 1,
                          f"images={images!r}", "plan"):
            continue
        image = images[0]
        out_w, out_h = case["out"]
        chk.expect_eq(f"{cid}.index", image.get("index"), 0, "plan")
        chk.expect_eq(f"{cid}.name", image.get("name"), name, "plan")
        chk.expect_eq(f"{cid}.origin", image.get("origin"),
                      {"width": case["w"], "height": case["h"]}, "plan")
        chk.expect_eq(f"{cid}.output", image.get("output"),
                      {"width": out_w, "height": out_h}, "plan")
        scale = image.get("scale")
        chk.expect(f"{cid}.scale",
                   isinstance(scale, (int, float)) and abs(float(scale) - case["scale"]) < 5e-6,
                   f"expected {round(case['scale'], 6)}, got {scale!r} ({case['why']})", "plan")
        chk.expect_eq(f"{cid}.mode", image.get("mode"), case["mode"], "plan")
        chk.expect_eq(f"{cid}.aligned", image.get("aligned"), case["aligned"], "plan")
        note = image.get("note")
        chk.expect(f"{cid}.note", isinstance(note, str) and len(note) > 0, f"note={note!r}",
                   "plan")

    # Batch request: order/index/name echo must be stable.
    batch = [dict(name="a.jpg", width=1280, height=720),
             dict(name="b.jpg", width=1080, height=1920),
             dict(name="c.jpg", width=1000, height=667)]
    resp = client.plan(batch, "1080p", align=8)
    if chk.expect_eq("plan.batch.http_status", resp.status_code, 200, "plan"):
        images = resp.json().get("images") or []
        chk.expect_eq("plan.batch.count", len(images), 3, "plan")
        expected_out = [{"width": 1920, "height": 1080}, {"width": 1080, "height": 1920},
                        {"width": 1616, "height": 1080}]
        for index, expected in enumerate(expected_out):
            if index < len(images):
                image = images[index]
                chk.expect_eq(f"plan.batch[{index}].index", image.get("index"), index, "plan")
                chk.expect_eq(f"plan.batch[{index}].name", image.get("name"),
                              batch[index]["name"], "plan")
                chk.expect_eq(f"plan.batch[{index}].output", image.get("output"), expected,
                              "plan")

    # Negative plan cases.
    bad = client.plan([{"name": "a.jpg", "width": 1280, "height": 720}], "foo")
    chk.expect_eq("plan.invalid_target.status", bad.status_code, 400, "plan")
    chk.expect("plan.invalid_target.detail",
               isinstance((bad.json() or {}).get("detail"), str) if bad.status_code == 400 else False,
               f"body={bad.text[:200]!r}", "plan")
    bad = client.plan([{"name": "a.jpg", "width": 0, "height": 720}], "1080p")
    chk.expect_eq("plan.zero_width.status", bad.status_code, 400, "plan")
    bad = client.plan([{"name": "a.jpg", "width": 1280, "height": 720}], "1080p", align=1)
    chk.expect_eq("plan.invalid_align.status", bad.status_code, 400, "plan")


# ---------------------------------------------------------------------------
# Section 3: negative API paths
# ---------------------------------------------------------------------------
def section_negative(chk: Checker, client: ApiClient, artifact_dir: Path) -> None:
    missing = "does-not-exist-qa"
    chk.expect_eq("negative.get_job_404.status",
                  client.get(f"/api/jobs/{missing}").status_code, 404, "negative")
    chk.expect_eq("negative.get_file_404.status",
                  client.get(f"/api/jobs/{missing}/files/0").status_code, 404, "negative")
    chk.expect_eq("negative.get_file_download_404.status",
                  client.get(f"/api/jobs/{missing}/files/0/download").status_code, 404,
                  "negative")
    chk.expect_eq("negative.delete_job_404.status",
                  client.delete_job(missing).status_code, 404, "negative")
    chk.expect_eq("negative.cancel_job_404.status",
                  client.cancel_job(missing).status_code, 404, "negative")
    chk.expect_eq("negative.zip_404.status",
                  client.get(f"/api/jobs/{missing}/download.zip").status_code, 404, "negative")

    sample = (SAMPLES_DIR / "photo_1280x720.jpg").read_bytes()
    # Unsupported file type -> 415 (docs/API.md 13).
    resp = client.post("/api/jobs", files={"files": ("evil.txt", b"not an image", "text/plain")},
                       data={"target": "1080p"})
    chk.expect_eq("negative.txt_415.status", resp.status_code, 415, "negative")
    # Corrupt image -> 400.
    resp = client.post("/api/jobs",
                       files={"files": ("corrupt.jpg", b"\x00\x01\x02not-a-jpeg", "image/jpeg")},
                       data={"target": "1080p"})
    chk.expect_eq("negative.corrupt_400.status", resp.status_code, 400, "negative")
    # Invalid target on job creation -> 400.
    resp = client.post("/api/jobs",
                       files={"files": ("ok.jpg", sample, "image/jpeg")},
                       data={"target": "foo"})
    chk.expect_eq("negative.job_invalid_target_400.status", resp.status_code, 400, "negative")
    # Missing target field -> 4xx (400 or FastAPI 422).
    resp = client.post("/api/jobs", files={"files": ("ok.jpg", sample, "image/jpeg")})
    chk.expect_in("negative.job_missing_target.status", resp.status_code, (400, 422), "negative")
    # Missing files field -> 4xx.
    resp = client.post("/api/jobs", data={"target": "1080p"})
    chk.expect_in("negative.job_missing_files.status", resp.status_code, (400, 422), "negative")


# ---------------------------------------------------------------------------
# Section 4: tile-seam / content-integrity probe on a near-uniform gradient
# ---------------------------------------------------------------------------
def section_tile(chk: Checker, client: ApiClient, artifact_dir: Path,
                 created_jobs: list[str], args) -> None:
    cid = "tile.gradient700"
    path = load_samples([GRADIENT_SAMPLE])[0]

    plan_resp = client.plan([{"name": GRADIENT_SAMPLE, "width": 700, "height": 700}], TIER)
    if chk.expect_eq(f"{cid}.plan_http_status", plan_resp.status_code, 200, "tile"):
        images = plan_resp.json().get("images") or []
        planned = tuple(images[0]["output"][k] for k in ("width", "height")) if images else None
        chk.expect_eq(f"{cid}.plan_output", planned, GRADIENT_OUT, "tile")

    resp = client.create_job([path], target=TIER, format="auto")
    if not chk.expect_eq(f"{cid}.job_create_status", resp.status_code, 202, "tile"):
        return
    job_id = resp.json().get("job_id")
    if not isinstance(job_id, str) or not job_id:
        chk.fail(f"{cid}.job_id", f"body={resp.text[:200]!r}", "tile")
        return
    created_jobs.append(job_id)
    try:
        final, wall, _samples = client.wait_job(job_id, timeout=args.job_timeout)
    except ApiError as exc:
        chk.fail(f"{cid}.job_done", str(exc), "tile")
        return
    if not chk.expect_eq(f"{cid}.job_terminal_status", final.get("status"), "done", "tile"):
        return

    entry = (final.get("files") or [{}])[0]
    chk.expect_eq(f"{cid}.output", entry.get("output"),
                  {"width": GRADIENT_OUT[0], "height": GRADIENT_OUT[1]}, "tile")
    chk.expect_eq(f"{cid}.output_format", entry.get("output_format"), "png", "tile")

    download = client.file_bytes(job_id, 0, download=True)
    if not chk.expect_eq(f"{cid}.download_status", download.status_code, 200, "tile"):
        return
    tile_dir = artifact_dir / "tile"
    tile_dir.mkdir(parents=True, exist_ok=True)
    (tile_dir / f"gradient_{TIER}.png").write_bytes(download.content)

    info = image_info(download.content)
    chk.expect_eq(f"{cid}.size", (info["width"], info["height"]), GRADIENT_OUT, "tile")
    chk.expect_eq(f"{cid}.format", info["format"], "PNG", "tile")
    problems = content_problems(info, reference=image_info(path.read_bytes()))
    chk.expect(f"{cid}.content_valid", not problems, "; ".join(problems), "tile")

    with Image.open(io.BytesIO(download.content)) as im:
        metrics = tile_metrics(im, grid=TILE_GRID, block=32)
    chk.expect(f"{cid}.tile_mean_uniform", metrics["max_cell_deviation"] < TILE_MEAN_TOL,
               f"global={metrics['global_mean']:.2f}, "
               f"max |tile-global|={metrics['max_cell_deviation']:.2f} "
               f"(tol {TILE_MEAN_TOL}), tile means="
               f"{[round(m, 2) for m in metrics['cell_means']]}", "tile")
    chk.expect(f"{cid}.block_mean_uniform", metrics["max_block_deviation"] < TILE_MEAN_TOL,
               f"{metrics['block_count']} blocks of 32x32px: "
               f"max |block-global|={metrics['max_block_deviation']:.2f} "
               f"(tol {TILE_MEAN_TOL})", "tile")
    chk.expect(f"{cid}.no_column_seam", metrics["column_jump_max"] < SEAM_TOL,
               f"max adjacent column-mean jump={metrics['column_jump_max']:.3f} "
               f"(median {metrics['column_jump_median']:.5f}, tol {SEAM_TOL})", "tile")
    chk.expect(f"{cid}.no_row_seam", metrics["row_jump_max"] < SEAM_TOL,
               f"max adjacent row-mean jump={metrics['row_jump_max']:.3f} "
               f"(median {metrics['row_jump_median']:.5f}, tol {SEAM_TOL})", "tile")
    chk.pass_(f"{cid}.job_wall_time", f"{wall:.2f}s")

    dresp = client.delete_job(job_id)
    chk.expect_eq(f"{cid}.delete", dresp.status_code, 200, "tile")


# ---------------------------------------------------------------------------
# Section 5: one real job per tier / special geometry -- produced bytes must
# match /api/plan exactly (Lead acceptance point: "4 档位实际输出像素").
# ---------------------------------------------------------------------------
def section_tier_outputs(chk: Checker, client: ApiClient, artifact_dir: Path,
                         created_jobs: list[str], args) -> None:
    out_dir = artifact_dir / "tier_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, (sample, tier, expected, pil_format) in enumerate(TIER_OUTPUT_CASES):
        cid = f"tiers[{index}].{Path(sample).stem}@{tier}"
        path = load_samples([sample])[0]
        with Image.open(path) as im:
            input_w, input_h = im.size

        plan_resp = client.plan([{"name": sample, "width": input_w, "height": input_h}], tier)
        planned = None
        if plan_resp.status_code == 200:
            images = plan_resp.json().get("images") or []
            if images:
                planned = images[0].get("output")
        chk.expect_eq(f"{cid}.plan_output", planned,
                      {"width": expected[0], "height": expected[1]}, "tiers")

        resp = client.create_job([path], target=tier, format="auto")
        if not chk.expect_eq(f"{cid}.create_status", resp.status_code, 202, "tiers"):
            continue
        job_id = resp.json().get("job_id")
        if not isinstance(job_id, str) or not job_id:
            chk.fail(f"{cid}.job_id", f"body={resp.text[:200]!r}", "tiers")
            continue
        created_jobs.append(job_id)
        try:
            final, wall, _samples = client.wait_job(job_id, timeout=args.job_timeout)
        except ApiError as exc:
            chk.fail(f"{cid}.job_done", str(exc), "tiers")
            continue
        if not chk.expect_eq(f"{cid}.terminal_status", final.get("status"), "done", "tiers"):
            continue
        entry = (final.get("files") or [{}])[0]
        chk.expect_eq(f"{cid}.job_output", entry.get("output"),
                      {"width": expected[0], "height": expected[1]}, "tiers")
        chk.expect_eq(f"{cid}.job_output_format", entry.get("output_format"), "jpg", "tiers")

        download = client.file_bytes(job_id, 0, download=True)
        if not chk.expect_eq(f"{cid}.download_status", download.status_code, 200, "tiers"):
            continue
        info = image_info(download.content)
        chk.expect_eq(f"{cid}.actual_size", (info["width"], info["height"]), expected, "tiers")
        chk.expect_eq(f"{cid}.actual_format", info["format"], pil_format, "tiers")
        problems = content_problems(info, reference=image_info(path.read_bytes()))
        chk.expect(f"{cid}.content_valid", not problems, "; ".join(problems), "tiers")
        (out_dir / f"{Path(sample).stem}_{tier}.jpg").write_bytes(download.content)
        chk.pass_(f"{cid}.wall_time", f"{wall:.2f}s")

        dresp = client.delete_job(job_id)
        chk.expect_eq(f"{cid}.delete", dresp.status_code, 200, "tiers")


# ---------------------------------------------------------------------------
# Section 6: main multi-file job
# ---------------------------------------------------------------------------
def section_job(chk: Checker, client: ApiClient, artifact_dir: Path, args,
                created_jobs: list[str]) -> tuple[dict, float]:
    paths = load_samples([name for name, *_ in JOB_FILES])
    originals = {}
    reference_stats = {}
    for name, *_ in JOB_FILES:
        with Image.open(SAMPLES_DIR / name) as im:
            originals[name] = (im.width, im.height)
        reference_stats[name] = image_info((SAMPLES_DIR / name).read_bytes())

    started = time.monotonic()
    resp = client.create_job(paths, target=TIER, format="auto")
    if not chk.expect_eq("job.create.http_status", resp.status_code, 202, "job"):
        chk.add("job.create.body", "FAIL", resp.text[:500], "job")
        raise ApiError(f"POST /api/jobs failed: {resp.status_code} {resp.text[:300]}", resp)
    created = resp.json()
    job_id = created.get("job_id")
    if not chk.expect("job.create.job_id", isinstance(job_id, str) and len(job_id) >= 6,
                      f"job_id={job_id!r}", "job"):
        raise ApiError("POST /api/jobs did not return a usable job_id")
    created_jobs.append(job_id)
    print(f"      job_id={job_id}", flush=True)

    chk.expect_in("job.create.status", created.get("status"), ("queued", "running"), "job")
    chk.expect("job.create.total", created.get("total") == len(JOB_FILES),
               f"total={created.get('total')!r}", "job")
    created_at = created.get("created_at")
    try:
        datetime.fromisoformat(str(created_at))
        created_ok = True
    except (TypeError, ValueError):
        created_ok = False
    chk.expect("job.create.created_at", created_ok, f"created_at={created_at!r}", "job")
    chk.expect("job.create.queue_position",
               isinstance(created.get("queue_position"), int) and created["queue_position"] >= 0,
               f"queue_position={created.get('queue_position')!r}", "job")

    target = created.get("target") or {}
    chk.expect_eq("job.create.target.id", target.get("id"), TIER, "job")
    chk.expect_eq("job.create.target.align", target.get("align"), ALIGN, "job")
    chk.expect_eq("job.create.target.value", target.get("value"), 1080, "job")

    engine = created.get("engine") or {}
    chk.expect("job.create.engine.backend", engine.get("backend") in ("realesrgan", "pil"),
               f"engine={engine!r}", "job")
    if args.expect_engine:
        chk.expect_eq("job.create.engine.expected_backend", engine.get("backend"),
                      args.expect_engine, "job")

    # v1.1 model echo in the job response (docs/API.md 14.4); default model=auto.
    job_model = created.get("model")
    if chk.expect("job.create.model_object", isinstance(job_model, dict),
                  f"model={job_model!r}", "job"):
        chk.expect_eq("job.create.model.id", job_model.get("id"), "auto", "job")
        chk.expect("job.create.model.label", isinstance(job_model.get("label"), str)
                   and bool(job_model["label"]), f"label={job_model.get('label')!r}", "job")
        chk.expect_eq("job.create.model.denoise", job_model.get("denoise"), False, "job")
        chk.expect_eq("job.create.model.denoise_level", job_model.get("denoise_level"), 0,
                      "job")

    files = created.get("files") or []
    chk.expect_eq("job.create.files_count", len(files), len(JOB_FILES), "job")
    for index, (name, out_size, api_fmt, pil_fmt, mime) in enumerate(JOB_FILES):
        if index >= len(files):
            chk.fail(f"job.create.files[{index}].present", "missing file entry", "job")
            continue
        entry = files[index]
        cid = f"job.create.files[{index}]"
        chk.expect_eq(f"{cid}.index", entry.get("index"), index, "job")
        chk.expect_eq(f"{cid}.name", entry.get("name"), name, "job")
        chk.expect_eq(f"{cid}.size", entry.get("size"), (SAMPLES_DIR / name).stat().st_size, "job")
        chk.expect_eq(f"{cid}.original", entry.get("original"),
                      {"width": originals[name][0], "height": originals[name][1]}, "job")
        chk.expect_eq(f"{cid}.output", entry.get("output"),
                      {"width": out_size[0], "height": out_size[1]}, "job")
        # The worker may pick up file 0 before the 202 response is serialised;
        # a fresh job is normally all-pending but running/0-progress is benign.
        status = entry.get("status")
        chk.expect_in(f"{cid}.create_status", status, ("pending", "running"), "job")
        progress = entry.get("progress")
        chk.expect(f"{cid}.create_progress",
                   isinstance(progress, (int, float)) and 0 <= progress <= 100,
                   f"progress={progress!r}", "job")
        if status != "pending":
            chk.warn(f"{cid}.create_status_pending",
                     f"status={status!r} in the 202 body (worker raced the response)", "job")
        chk.expect_eq(f"{cid}.preview_url", entry.get("preview_url"),
                      f"/api/jobs/{job_id}/files/{index}", "job")
        chk.expect_eq(f"{cid}.download_url", entry.get("download_url"),
                      f"/api/jobs/{job_id}/files/{index}/download", "job")
        chk.expect_eq(f"{cid}.model", entry.get("model"), "auto", "job")
        chk.expect("job.create.files.model_used_key", "model_used" in entry,
                   f"keys={sorted(entry)}", "job")
        chk.expect("job.create.files.denoise_level_key", "denoise_level" in entry,
                   f"keys={sorted(entry)}", "job")

    # A result that is not ready must answer 404/409, never serve bytes.
    early = client.file_bytes(job_id, 0, download=False)
    if early.status_code == 200:
        chk.warn("job.preview_before_ready",
                 "result already available on the very first poll (engine faster than harness?)",
                 "job")
    else:
        chk.expect_in("job.preview_before_ready", early.status_code, (404, 409), "job")

    # ---- poll to completion -------------------------------------------
    samples: list[dict] = []
    try:
        final, poll_elapsed, samples = client.wait_job(
            job_id, timeout=args.job_timeout, interval=0.5,
            on_sample=samples.append)
    except ApiError as exc:
        chk.fail("job.poll.completes", str(exc), "job")
        client.cancel_job(job_id)
        time.sleep(1.0)
        raise
    wall = time.monotonic() - started
    print(f"      job finished in {wall:.2f}s wall, {len(samples)} polls, "
          f"status={final.get('status')!r}, job.elapsed_sec={final.get('elapsed_sec')!r}",
          flush=True)

    # Progress must never go backwards on the task level.
    max_progress = -1.0
    regression = None
    for snapshot in samples:
        progress = snapshot.get("progress")
        if not isinstance(progress, (int, float)):
            continue
        if progress + 0.5 < max_progress:
            regression = (max_progress, progress)
        max_progress = max(max_progress, float(progress))
    chk.expect("job.poll.progress_monotonic", regression is None,
               "" if regression is None else f"progress went {regression[0]} -> {regression[1]}",
               "job")
    if samples:
        chk.expect("job.poll.saw_running",
                   any(s.get("status") == "running" for s in samples),
                   f"statuses seen={sorted({s.get('status') for s in samples})}", "job")

    # ---- terminal job state -------------------------------------------
    if final.get("status") == "error":
        errors = [f"file {f.get('index')}: {f.get('error')}" for f in final.get("files", [])
                  if f.get("error")]
        chk.add("job.terminal.status", "FAIL",
                f"job status=error, error={final.get('error')!r}, files={errors}", "job")
        return final, wall
    chk.expect_eq("job.terminal.status", final.get("status"), "done", "job")
    chk.expect_eq("job.terminal.error", final.get("error"), None, "job")
    chk.expect("job.terminal.total", final.get("total") == len(JOB_FILES),
               f"total={final.get('total')!r}", "job")
    finished_at = final.get("finished_at")
    try:
        datetime.fromisoformat(str(finished_at))
        finished_ok = True
    except (TypeError, ValueError):
        finished_ok = False
    chk.expect("job.terminal.finished_at", finished_ok, f"finished_at={finished_at!r}", "job")
    chk.expect("job.terminal.started_at",
               isinstance(final.get("started_at"), str) and bool(final.get("started_at")),
               f"started_at={final.get('started_at')!r}", "job")
    progress = final.get("progress")
    chk.expect("job.terminal.progress_100",
               isinstance(progress, (int, float)) and abs(float(progress) - 100.0) < 0.05,
               f"progress={progress!r}", "job")
    elapsed = final.get("elapsed_sec")
    chk.expect("job.terminal.elapsed_sec",
               isinstance(elapsed, (int, float)) and 0 <= elapsed < args.job_timeout,
               f"elapsed_sec={elapsed!r}", "job")
    if isinstance(elapsed, (int, float)):
        drift = abs(float(elapsed) - wall)
        if drift > max(5.0, wall * 0.2):
            chk.warn("job.terminal.elapsed_sec_consistent",
                     f"elapsed_sec={elapsed:.2f} vs wall={wall:.2f}", "job")
        else:
            chk.pass_("job.terminal.elapsed_sec_consistent",
                      f"elapsed_sec={elapsed:.2f} ~ wall={wall:.2f}", "job")

    job_files = final.get("files") or []
    chk.expect_eq("job.terminal.files_count", len(job_files), len(JOB_FILES), "job")
    for index, (name, out_size, api_fmt, pil_fmt, mime) in enumerate(JOB_FILES):
        if index >= len(job_files):
            continue
        entry = job_files[index]
        cid = f"job.terminal.files[{index}]"
        chk.expect_eq(f"{cid}.status", entry.get("status"), "done", "job")
        chk.expect_eq(f"{cid}.progress", entry.get("progress"), 100, "job")
        chk.expect_eq(f"{cid}.error", entry.get("error"), None, "job")
        chk.expect_eq(f"{cid}.output_format", entry.get("output_format"), api_fmt, "job")
        size = entry.get("output_size")
        chk.expect(f"{cid}.output_size", isinstance(size, int) and size > 0,
                   f"output_size={size!r}", "job")
        chk.expect_eq(f"{cid}.output", entry.get("output"),
                      {"width": out_size[0], "height": out_size[1]}, "job")
        chk.expect_eq(f"{cid}.model", entry.get("model"), "auto", "job")
        chk.expect_eq(f"{cid}.denoise_level", entry.get("denoise_level"), 0, "job")
        used_ok, used_detail = model_used_matches("auto", entry.get("model_used"))
        chk.expect(f"{cid}.model_used", used_ok, used_detail, "job")

    # A finished job must reject cancel with 409 (docs/API.md 9).
    cancel_done = client.cancel_job(job_id)
    chk.expect_eq("job.cancel_after_done_409", cancel_done.status_code, 409, "job")

    # health cache counters must reflect the live job.
    health_after = client.health()
    cache_after = health_after.get("cache") or {}
    chk.expect("health.cache.jobs_ge_1", cache_after.get("jobs", 0) >= 1,
               f"cache={cache_after!r}", "health")
    chk.expect("health.cache.result_mb_positive", float(cache_after.get("result_mb") or 0) > 0,
               f"cache={cache_after!r}", "health")

    # ---- downloads -----------------------------------------------------
    downloads_dir = artifact_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    preview_bytes: dict[int, bytes] = {}
    for index, (name, out_size, api_fmt, pil_fmt, mime) in enumerate(JOB_FILES):
        if index >= len(job_files):
            continue
        entry = job_files[index]
        cid = f"job.download[{index}]"
        expected_name = expected_result_name(name, TIER, "jpg" if api_fmt == "jpg" else "png")

        preview = client.file_bytes(job_id, index)
        if not chk.expect_eq(f"{cid}.preview_http_status", preview.status_code, 200, "job"):
            continue
        content_type = preview.headers.get("content-type", "").split(";")[0].strip().lower()
        chk.expect_eq(f"{cid}.preview_content_type", content_type, mime, "job")
        chk.expect("job.downloads.cache_control",
                   "no-cache" in preview.headers.get("cache-control", "").lower(),
                   f"cache-control={preview.headers.get('cache-control')!r}", "job")
        try:
            info = image_info(preview.content)
        except Exception as exc:  # noqa: BLE001
            chk.fail(f"{cid}.preview_pil_opens", f"{type(exc).__name__}: {exc}", "job")
            continue
        preview_bytes[index] = preview.content
        chk.expect_eq(f"{cid}.preview_size", (info["width"], info["height"]), out_size, "job")
        chk.expect_eq(f"{cid}.preview_format", info["format"], pil_fmt, "job")
        problems = content_problems(info, reference=reference_stats[name])
        chk.expect(f"{cid}.preview_content_valid", not problems, "; ".join(problems), "job")
        if isinstance(entry.get("output_size"), int):
            chk.expect_eq(f"{cid}.output_size_matches_bytes", len(preview.content),
                          entry["output_size"], "job")
        if api_fmt == "png" and name.startswith("alpha_"):
            chk.expect_eq(f"{cid}.alpha_mode", info["mode"], "RGBA", "job")
            chk.expect(f"{cid}.alpha_preserved",
                       info["alpha_min"] < 32 and info["alpha_max"] > 240
                       and info["alpha_transparent_ratio"] > 0.03,
                       f"alpha_min={info['alpha_min']:.0f} alpha_max={info['alpha_max']:.0f} "
                       f"transparent={info['alpha_transparent_ratio']:.1%}", "job")
        (downloads_dir / f"{index}_{expected_name}").write_bytes(preview.content)

        download = client.file_bytes(job_id, index, download=True)
        if not chk.expect_eq(f"{cid}.download_http_status", download.status_code, 200, "job"):
            continue
        dl_type = download.headers.get("content-type", "").split(";")[0].strip().lower()
        chk.expect_eq(f"{cid}.download_content_type", dl_type, mime, "job")
        displacement = download.headers.get("content-disposition", "")
        parsed = parse_content_disposition(displacement)
        chk.expect_eq(f"{cid}.download_disposition_type", parsed.get("_type"), "attachment", "job")
        chk.expect_eq(f"{cid}.download_filename",
                      content_disposition_filename(displacement), expected_name, "job")
        chk.expect(f"{cid}.download_bytes_equal_preview",
                   download.content == preview_bytes.get(index), "", "job")
        (downloads_dir / f"{index}_{expected_name}.attachment").write_bytes(download.content)

    # Custom ``?filename=`` on the first result (docs/API.md 8).
    custom = client.file_bytes(job_id, 0, download=True,
                               params={"filename": "qa_custom_save_name.jpg"})
    if chk.expect_eq("job.custom_filename.status", custom.status_code, 200, "job"):
        custom_name = content_disposition_filename(custom.headers.get("content-disposition"))
        chk.expect("job.custom_filename.applied",
                   isinstance(custom_name, str) and custom_name.startswith("qa_custom_save_name")
                   and custom_name.endswith(".jpg"),
                   f"Content-Disposition filename={custom_name!r}", "job")

    # ---- zip -----------------------------------------------------------
    expected_zip_names = {expected_result_name(name, TIER, "jpg" if api_fmt == "jpg" else "png")
                          for name, _out, api_fmt, _pil, _mime in JOB_FILES}
    expected_zip_filename = f"PicPolish_{str(job_id)[:8]}.zip"
    zresp = client.zip_bytes(job_id)
    if chk.expect_eq("job.zip.http_status", zresp.status_code, 200, "job"):
        ztype = zresp.headers.get("content-type", "").split(";")[0].strip().lower()
        chk.expect_eq("job.zip.content_type", ztype, "application/zip", "job")
        chk.expect_eq("job.zip.filename",
                      content_disposition_filename(zresp.headers.get("content-disposition")),
                      expected_zip_filename, "job")
        try:
            with zipfile.ZipFile(io.BytesIO(zresp.content)) as archive:
                bad = archive.testzip()
                chk.expect("job.zip.integrity", bad is None, f"corrupt entry: {bad}", "job")
                names = archive.namelist()
                chk.expect_eq("job.zip.entries", set(names), expected_zip_names, "job")
                for index, name in enumerate(JOB_FILES):
                    entry_name = expected_result_name(name[0], TIER,
                                                      "jpg" if name[2] == "jpg" else "png")
                    if entry_name not in names or index not in preview_bytes:
                        continue
                    chk.expect(f"job.zip.entry[{index}].bytes",
                               archive.read(entry_name) == preview_bytes[index],
                               f"zip entry {entry_name} matches downloaded result", "job")
        except zipfile.BadZipFile as exc:
            chk.fail("job.zip.opens", f"BadZipFile: {exc}", "job")
        (downloads_dir / "results.zip").write_bytes(zresp.content)
    else:
        chk.add("job.zip.opens", "FAIL", f"zip download failed: {zresp.status_code}", "job")

    # ---- server-local cache layout ------------------------------------
    if args.check_cache and CACHE_DIR.is_dir():
        upload_dir = CACHE_DIR / "uploads" / str(job_id)
        result_dir = CACHE_DIR / "results" / str(job_id)
        chk.expect("job.cache.uploads_dir", upload_dir.is_dir(),
                   f"{upload_dir} exists={upload_dir.is_dir()}", "cache")
        chk.expect("job.cache.results_dir", result_dir.is_dir(),
                   f"{result_dir} exists={result_dir.is_dir()}", "cache")
        upload_files = [p for p in upload_dir.rglob("*") if p.is_file()] if upload_dir.is_dir() else []
        result_files = [p for p in result_dir.rglob("*") if p.is_file()] if result_dir.is_dir() else []
        chk.expect_eq("job.cache.uploads_count", len(upload_files), len(JOB_FILES), "cache")
        chk.expect_eq("job.cache.results_count", len(result_files), len(JOB_FILES), "cache")
        result_names = {p.name for p in result_files}
        chk.expect_eq("job.cache.result_names", result_names, expected_zip_names, "cache")
        total_result_bytes = sum(p.stat().st_size for p in result_files)
        expected_result_bytes = sum(f.get("output_size") or 0 for f in job_files)
        if total_result_bytes != expected_result_bytes:
            chk.warn("job.cache.result_sizes",
                     f"results dir {total_result_bytes}B != sum(output_size) "
                     f"{expected_result_bytes}B", "cache")
        upload_bytes = sum(p.stat().st_size for p in upload_files)
        expected_upload_bytes = sum((SAMPLES_DIR / n).stat().st_size for n, *_ in JOB_FILES)
        if upload_bytes != expected_upload_bytes:
            chk.warn("job.cache.upload_sizes",
                     f"uploads dir {upload_bytes}B != uploaded {expected_upload_bytes}B", "cache")
        expected_freed = upload_bytes + total_result_bytes

        # ---- DELETE ---------------------------------------------------
        dresp = client.delete_job(job_id)
        if chk.expect_eq("job.delete.http_status", dresp.status_code, 200, "job"):
            body = dresp.json()
            chk.expect_eq("job.delete.deleted", body.get("deleted"), True, "job")
            chk.expect_eq("job.delete.job_id", body.get("job_id"), job_id, "job")
            freed = body.get("freed_bytes")
            chk.expect("job.delete.freed_bytes",
                       isinstance(freed, int) and freed > 0, f"freed_bytes={freed!r}", "job")
            if isinstance(freed, int):
                if abs(freed - expected_freed) <= max(1024, expected_freed * 0.01):
                    chk.pass_("job.delete.freed_bytes_exact",
                              f"freed_bytes={freed} ~ expected {expected_freed}", "job")
                else:
                    chk.warn("job.delete.freed_bytes_exact",
                             f"freed_bytes={freed} vs expected {expected_freed}", "job")
        chk.expect_eq("job.delete.get_after_status",
                      client.get(f"/api/jobs/{job_id}").status_code, 404, "job")
        chk.expect("job.delete.cache_dirs_removed",
                   not upload_dir.exists() and not result_dir.exists(),
                   f"uploads exists={upload_dir.exists()}, results exists={result_dir.exists()}",
                   "cache")
    else:
        chk.skip("job.cache.layout", f"cache dir not present: {CACHE_DIR}", "cache")
        dresp = client.delete_job(job_id)
        chk.expect_eq("job.delete.http_status", dresp.status_code, 200, "job")

    return final, wall


# ---------------------------------------------------------------------------
# Section 5: cancelling a fresh job
# ---------------------------------------------------------------------------
def section_cancel(chk: Checker, client: ApiClient, created_jobs: list[str]) -> None:
    tiny = SAMPLES_DIR / "tiny_320x240.png"
    if not tiny.is_file():
        chk.skip("cancel.submit", f"missing sample {tiny}", "cancel")
        return
    resp = client.create_job([tiny, tiny, tiny], target="720p")
    if not chk.expect_eq("cancel.submit.http_status", resp.status_code, 202, "cancel"):
        return
    job_id = resp.json().get("job_id")
    if not isinstance(job_id, str) or not job_id:
        chk.fail("cancel.submit.job_id", f"body={resp.text[:300]!r}", "cancel")
        return
    created_jobs.append(job_id)

    cancel = client.cancel_job(job_id)
    if cancel.status_code == 409:
        chk.warn("cancel.http_status",
                 "job finished before the cancel request arrived (too fast to exercise cancelling)",
                 "cancel")
        try:
            client.wait_job(job_id, timeout=60)
        except ApiError:
            pass
        chk.expect_eq("cancel.delete_after_cancel", client.delete_job(job_id).status_code, 200,
                      "cancel")
        return
    if not chk.expect_eq("cancel.http_status", cancel.status_code, 200, "cancel"):
        return
    body = cancel.json()
    chk.expect_eq("cancel.response.job_id", body.get("job_id"), job_id, "cancel")
    chk.expect_eq("cancel.response.status", body.get("status"), "cancelled", "cancel")

    try:
        final, _elapsed, _samples = client.wait_job(job_id, timeout=60)
    except ApiError as exc:
        chk.fail("cancel.terminal", str(exc), "cancel")
        return
    chk.expect_eq("cancel.terminal.status", final.get("status"), "cancelled", "cancel")

    # ``cancel()`` flips the job status immediately while the single worker may
    # still be finishing the in-flight file (``processing=True``), so wait for
    # the file states to settle before asserting / deleting.
    settled = final
    settle_started = time.monotonic()
    while time.monotonic() - settle_started < 30.0:
        statuses = {f.get("status") for f in settled.get("files", [])}
        if not (statuses & {"pending", "running"}):
            break
        time.sleep(0.2)
        settled = client.job(job_id)
    settle_sec = time.monotonic() - settle_started
    chk.pass_("cancel.worker_settled", f"after {settle_sec:.2f}s")
    file_statuses = {f.get("status") for f in settled.get("files", [])}
    chk.expect("cancel.terminal.no_active_files", file_statuses <= {"cancelled", "done"},
               f"file statuses={sorted(file_statuses)}", "cancel")

    # DELETE is allowed once the worker stops touching the job; retry briefly
    # when it still answers 409 for the settling window.
    delete_resp = client.delete_job(job_id)
    delete_wait = 0.0
    while delete_resp.status_code == 409 and delete_wait < 30.0:
        time.sleep(0.2)
        delete_wait += 0.2
        delete_resp = client.delete_job(job_id)
    if delete_resp.status_code == 200 and delete_wait > 0.05:
        chk.pass_("cancel.delete_after_cancel",
                  f"deleted after {delete_wait:.2f}s settle (409 while worker finishing)")
    else:
        chk.expect_eq("cancel.delete_after_cancel", delete_resp.status_code, 200, "cancel")


# ---------------------------------------------------------------------------
# Section 6: the web UI is served (front-end deliverable)
# ---------------------------------------------------------------------------
def section_ui(chk: Checker, client: ApiClient) -> None:
    root = client.get("/")
    if chk.expect_eq("ui.root.http_status", root.status_code, 200, "ui"):
        html = root.text
        content_type = root.headers.get("content-type", "").lower()
        chk.expect("ui.root.content_type", "text/html" in content_type,
                   f"content-type={content_type!r}", "ui")
        chk.expect("ui.root.html_title", "picpolish" in html.lower(),
                   f"len={len(html)}", "ui")
        chk.expect("ui.root.has_app_js", "app.js" in html,
                   "index.html should reference web/app.js", "ui")
    index = client.get("/web/index.html")
    if chk.expect_eq("ui.index.http_status", index.status_code, 200, "ui"):
        chk.expect("ui.index.html_markup", "<html" in index.text.lower(), "", "ui")
    for asset in ("/web/app.js", "/web/style.css"):
        resp = client.get(asset)
        safe = asset.replace("/", ".").replace(".web.", ".")
        chk.expect_eq(f"ui.asset{safe}.http_status", resp.status_code, 200, "ui")
        if resp.status_code == 200:
            chk.expect(f"ui.asset{safe}.non_empty", len(resp.content) > 0,
                       f"bytes={len(resp.content)}", "ui")


# ---------------------------------------------------------------------------
# Offline sanity check of the expected plan table against app/target.py.
# ---------------------------------------------------------------------------
def self_test_plan() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from app.target import InvalidTarget, plan_target  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: cannot import app.target: {type(exc).__name__}: {exc}")
        return 2
    failures = 0
    for case in PLAN_CASES:
        align = case.get("align", ALIGN)
        label = f"{case['w']}x{case['h']}@{case['tier']}" + (
            f"/align={align}" if align != ALIGN else "")
        try:
            plan = plan_target(case["w"], case["h"], case["tier"], align=align)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {label}: raised {type(exc).__name__}: {exc}")
            failures += 1
            continue
        got = (plan["output"]["width"], plan["output"]["height"])
        problems = []
        if got != case["out"]:
            problems.append(f"output {got} != {case['out']}")
        if abs(plan["scale"] - case["scale"]) > 5e-6:
            problems.append(f"scale {plan['scale']} != {case['scale']:.6f}")
        if plan["mode"] != case["mode"]:
            problems.append(f"mode {plan['mode']} != {case['mode']}")
        if plan["aligned"] != case["aligned"]:
            problems.append(f"aligned {plan['aligned']} != {case['aligned']}")
        if problems:
            print(f"FAIL {label}: " + "; ".join(problems))
            failures += 1
        else:
            print(f"PASS {label}: {got} {plan['mode']}")
    for width, height, tier in ((0, 720, "1080p"), (1280, 720, "foo"), (1280, 8, "1080p")):
        try:
            plan_target(width, height, tier)
        except InvalidTarget:
            print(f"PASS invalid {width}x{height}@{tier}: InvalidTarget")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL invalid {width}x{height}@{tier}: wrong exception {type(exc).__name__}")
            failures += 1
        else:
            print(f"FAIL invalid {width}x{height}@{tier}: did not raise InvalidTarget")
            failures += 1
    print(f"SELF-TEST-PLAN: {'OK' if failures == 0 else str(failures) + ' FAIL'} "
          f"({len(PLAN_CASES)} cases)")
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
ALL_SECTIONS = ("health", "targets", "models", "plan", "negative", "tile", "tiers", "job",
                "cancel", "ui")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PicPolish Windows E2E test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8796",
                        help="running PicPolish server URL (default: %(default)s)")
    parser.add_argument("--artifact-dir", type=Path, default=None,
                        help="where to write downloads/JSON (default: fresh temp dir)")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="JSON summary path (default: <artifact-dir>/e2e_windows.json)")
    parser.add_argument("--job-timeout", type=float, default=300.0,
                        help="max seconds to wait for the multi-file job (default: %(default)s)")
    parser.add_argument("--expect-engine", default="",
                        help="fail unless engine.backend equals this (e.g. realesrgan)")
    parser.add_argument("--expect-device", default="",
                        help="fail unless engine.device equals this (e.g. cuda)")
    parser.add_argument("--expect-fresh", action="store_true",
                        help="server must report an empty cache (freshly started)")
    parser.add_argument("--no-check-cache", dest="check_cache", action="store_false",
                        help="skip server-local cache/uploads|results layout checks")
    parser.add_argument("--sections", default=",".join(ALL_SECTIONS),
                        help=f"comma-separated subset of {','.join(ALL_SECTIONS)}")
    parser.add_argument("--self-test-plan", action="store_true",
                        help="offline: validate the expected plan table against app/target.py")
    args = parser.parse_args(argv)

    if args.self_test_plan:
        return self_test_plan()

    setup_console()
    wanted = {name.strip() for name in args.sections.split(",") if name.strip()}
    unknown = wanted - set(ALL_SECTIONS)
    if unknown:
        print(f"unknown sections: {sorted(unknown)}")
        return 2
    artifact_dir = (args.artifact_dir or Path(tempfile.mkdtemp(prefix="picpolish_e2e_"))).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"PicPolish Windows E2E  base_url={args.base_url}")
    print(f"artifacts: {artifact_dir}")
    print(f"sections : {','.join(name for name in ALL_SECTIONS if name in wanted)}")
    print("=" * 78)
    chk = Checker()
    client = ApiClient(args.base_url)
    created_jobs: list[str] = []
    job_holder: dict = {}
    models_holder: dict = {}
    health: dict = {}
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started = time.monotonic()
    exit_code = 0
    try:
        # Connectivity is fatal: there is nothing to test without a server.
        try:
            health = section_health(chk, client, args)
        except Exception as exc:  # noqa: BLE001
            print(f"FATAL: {args.base_url} is not a usable PicPolish server: "
                  f"{type(exc).__name__}: {exc}")
            return 2
        if "targets" in wanted:
            guarded(chk, "targets", lambda: section_targets(chk, client))
        if "models" in wanted:
            def run_models_api() -> None:
                models_holder.update(section_models_api(chk, client, artifact_dir))
            guarded(chk, "models", run_models_api)
        if "plan" in wanted:
            guarded(chk, "plan", lambda: section_plan(chk, client))
        if "negative" in wanted:
            guarded(chk, "negative", lambda: section_negative(chk, client, artifact_dir))
        if "tile" in wanted:
            guarded(chk, "tile", lambda: section_tile(chk, client, artifact_dir,
                                                      created_jobs, args))
        if "tiers" in wanted:
            guarded(chk, "tiers", lambda: section_tier_outputs(chk, client, artifact_dir,
                                                               created_jobs, args))
        if "job" in wanted:
            def run_job() -> None:
                final, wall = section_job(chk, client, artifact_dir, args, created_jobs)
                job_holder.update({
                    "job_id": final.get("job_id"),
                    "status": final.get("status"),
                    "wall_sec": round(wall, 3),
                    "elapsed_sec": final.get("elapsed_sec"),
                    "total": final.get("total"),
                })
            guarded(chk, "job", run_job)
        if "cancel" in wanted:
            guarded(chk, "cancel", lambda: section_cancel(chk, client, created_jobs))
        if "ui" in wanted:
            guarded(chk, "ui", lambda: section_ui(chk, client))
    finally:
        for job_id in created_jobs:
            try:
                client.delete_job(job_id)
            except Exception:  # noqa: BLE001 - cleanup must never mask the results
                pass
        client.close()
        duration = time.monotonic() - started
        summary = chk.summary()
        payload = {
            "suite": "windows-e2e",
            "base_url": args.base_url,
            "started_at": started_at,
            "duration_sec": round(duration, 3),
            "expect_engine": args.expect_engine or None,
            "expect_device": args.expect_device or None,
            "engine": (health or {}).get("engine"),
            "models_health": (health or {}).get("models"),
            "models_api": models_holder.get("models"),
            "job": job_holder,
            "summary": summary,
            "checks": chk.to_json(),
            "artifacts": str(artifact_dir),
        }
        out = args.json_out or (artifact_dir / "e2e_windows.json")
        write_json(out, payload)
        print("-" * 78)
        print(f"JSON summary: {out}")
        print(f"E2E SUMMARY: PASS={summary['pass']} FAIL={summary['fail']} "
              f"WARN={summary['warn']} SKIP={summary['skip']} elapsed={duration:.1f}s")
        if summary["failed_ids"]:
            print("FAILED CHECKS: " + ", ".join(summary["failed_ids"]))
        exit_code = 0 if summary["fail"] == 0 else 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
