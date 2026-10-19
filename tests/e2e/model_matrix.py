#!/usr/bin/env python
"""PicPolish R2 model matrix: multi-model + denoise acceptance (real GPU jobs).

Covers ``docs/API.md`` sections 6/7/14 and ``docs/CONTRACT.md`` sections 5/6:

1. ``--models`` sanity: the models the caller expects are all reported available
   (the full ``/api/models`` contract is asserted by ``e2e_windows.py``).
2. **Per-model job (720p -> 1080p)**: for auto / realesrgan-x4plus /
   realesrgan-x2plus / realesrgan-anime6b / realcugan /
   realcugan-conservative, submit one real job on a photo, and assert:
   exact output size/format, ``model`` echo, correct ``model_used``,
   content validity (black ratio etc.), and record elapsed time.
3. **Per-model 4K job** (unless ``--no-4k``): 1280x720 -> 3840x2160, exact size,
   content validity, elapsed time (contract: 4K within 180 s).
4. **realcugan denoise_level 0/1/2/3** on the Gaussian-noise sample: every level
   is a legal image, and level 3 differs measurably from level 0.
5. **anime6b vs x4plus** on the anime-style sample: the two models must produce
   measurably different output (proves the anime model really is used).
6. **rrdb denoise on/off** on the Gaussian-noise sample: outputs differ and the
   denoised result has less high-frequency noise energy.

Exit code 0 iff every blocking check passes.  Never starts/stops the server;
``e2e_windows.ps1`` owns the lifecycle.  Usage::

    .venv\\Scripts\\python.exe tests\\e2e\\model_matrix.py --base-url http://127.0.0.1:8796
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    MODEL_IDS,
    MODELS_EXPECTED,
    SAMPLES_DIR,
    ApiClient,
    Checker,
    content_problems,
    image_info,
    laplacian_variance,
    load_samples,
    mean_abs_diff,
    model_used_matches,
    rmse,
    setup_console,
    write_json,
)

TARGET_720P = "1080p"
TARGET_4K = "4k"
SIZE_1080P = (1920, 1080)
SIZE_4K = (3840, 2160)

# Model matrix uses a real photo (photo_1280x720.jpg) so every model gets the
# same bright, detailed input; model-specific content tests use the probes below.
MATRIX_SAMPLE = "photo_1280x720.jpg"
ANIME_SAMPLE = "anime_1280x720.png"
NOISY_SAMPLE = "noisy_gauss_1280x720.jpg"
DENOISE_LEVELS = (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def open_image(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return im.copy()


def sample_reference(name: str) -> dict:
    return image_info((SAMPLES_DIR / name).read_bytes())


def run_single_job(chk: Checker, client: ApiClient, artifact_dir: Path, cid: str,
                   sample: str, target: str, expected_size: tuple[int, int],
                   expected_pil_format: str, expected_api_format: str,
                   form: dict, *, job_timeout: float,
                   expected_denoise_level: int | None = None,
                   used_hints: dict | None = None) -> dict | None:
    """Submit one job, verify every documented field/size/content, return result."""
    path = load_samples([sample])[0]
    reference = sample_reference(sample)
    requested_model = str(form.get("model", "auto"))
    category = "matrix"
    resp = client.create_job([path], target=target, **form)
    if not chk.expect_eq(f"{cid}.create_status", resp.status_code, 202, category):
        print(f"      create body: {resp.text[:400]}", flush=True)
        return None
    created = resp.json()
    job_id = created.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        chk.fail(f"{cid}.job_id", f"body={created!r}", category)
        return None

    # Job-level model echo (docs/API.md 14.4).
    job_model = created.get("model")
    if chk.expect(f"{cid}.job.model_object", isinstance(job_model, dict),
                  f"model={job_model!r}", category):
        chk.expect_eq(f"{cid}.job.model.id", job_model.get("id"), requested_model, category)
        chk.expect(f"{cid}.job.model.label",
                   isinstance(job_model.get("label"), str) and bool(job_model["label"]),
                   f"label={job_model.get('label')!r}", category)
        expected_denoise = bool(form.get("denoise", False))
        chk.expect_eq(f"{cid}.job.model.denoise", job_model.get("denoise"), expected_denoise,
                      category)

    try:
        final, wall, _samples = client.wait_job(job_id, timeout=job_timeout)
    except Exception as exc:  # noqa: BLE001 - record and move on to the next model
        chk.fail(f"{cid}.job_done", f"{type(exc).__name__}: {exc}", category)
        try:
            client.delete_job(job_id)
        except Exception:  # noqa: BLE001
            pass
        return None
    elapsed = final.get("elapsed_sec")
    if not chk.expect_eq(f"{cid}.terminal_status", final.get("status"), "done", category):
        errors = [f.get("error") for f in final.get("files", []) if f.get("error")]
        chk.fail(f"{cid}.terminal_error", f"job.error={final.get('error')!r} files={errors}",
                 category)
        try:
            client.delete_job(job_id)
        except Exception:  # noqa: BLE001
            pass
        return None

    entry = (final.get("files") or [{}])[0]
    chk.expect_eq(f"{cid}.output", entry.get("output"),
                  {"width": expected_size[0], "height": expected_size[1]}, category)
    chk.expect_eq(f"{cid}.output_format", entry.get("output_format"), expected_api_format,
                  category)
    chk.expect_eq(f"{cid}.model", entry.get("model"), requested_model, category)
    used_ok, used_detail = model_used_matches(requested_model, entry.get("model_used"),
                                              hints=used_hints)
    chk.expect(f"{cid}.model_used", used_ok, used_detail, category)
    denoise_level = entry.get("denoise_level")
    chk.expect(f"{cid}.denoise_level_int", isinstance(denoise_level, int),
               f"denoise_level={denoise_level!r}", category)
    if expected_denoise_level is not None:
        chk.expect_eq(f"{cid}.denoise_level", denoise_level, expected_denoise_level, category)

    download = client.file_bytes(job_id, 0, download=True)
    if not chk.expect_eq(f"{cid}.download_status", download.status_code, 200, category):
        client.delete_job(job_id)
        return None
    info = image_info(download.content)
    chk.expect_eq(f"{cid}.actual_size", (info["width"], info["height"]), expected_size,
                  category)
    chk.expect_eq(f"{cid}.actual_format", info["format"], expected_pil_format, category)
    problems = content_problems(info, reference=reference)
    chk.expect(f"{cid}.content_valid", not problems, "; ".join(problems), category)

    out_dir = artifact_dir / "model_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "png" if expected_pil_format == "PNG" else "jpg"
    out_path = out_dir / f"{cid}.{suffix}"
    out_path.write_bytes(download.content)

    result = {
        "cid": cid,
        "job_id": job_id,
        "sample": sample,
        "target": target,
        "model": requested_model,
        "model_used": entry.get("model_used"),
        "denoise": bool(form.get("denoise", False)),
        "denoise_level": denoise_level,
        "size": list(expected_size),
        "format": info["format"],
        "bytes": len(download.content),
        "wall_sec": round(wall, 3),
        "elapsed_sec": elapsed,
        "black_ratio": round(info["black_ratio"], 5),
        "mean": round(info["mean"], 2),
        "image": open_image(download.content),
    }
    try:
        client.delete_job(job_id)
    except Exception:  # noqa: BLE001
        pass
    return result


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def save_panel(left_path: Path, right_path: Path, left_label: str, right_label: str,
               out_path: Path, crop: tuple[int, int] = (520, 300)) -> Path:
    """Side-by-side centre-crop of two model outputs (report evidence)."""
    from PIL import ImageDraw, ImageFont

    def crop_center(im: Image.Image, w: int, h: int) -> Image.Image:
        x = max(0, (im.width - w) // 2)
        y = max(0, (im.height - h) // 2)
        return im.crop((x, y, x + w, y + h))

    with Image.open(left_path) as left, Image.open(right_path) as right:
        left_im = crop_center(left.convert("RGB"), *crop)
        right_im = crop_center(right.convert("RGB"), *crop)
    cw, ch = crop
    panel = Image.new("RGB", (cw * 2 + 12, ch + 36), (24, 24, 24))
    panel.paste(left_im, (0, 36))
    panel.paste(right_im, (cw + 12, 36))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype(str(Path(r"C:\Windows\Fonts\segoeui.ttf")), 20)
    except Exception:  # noqa: BLE001 - font availability is best-effort
        font = ImageFont.load_default()
    draw.text((8, 8), left_label, font=font, fill=(235, 235, 235))
    draw.text((cw + 20, 8), right_label, font=font, fill=(120, 255, 160))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path, format="PNG", optimize=True)
    return out_path


def build_comparisons(artifact_dir: Path, denoise_results: dict[int, dict],
                      anime_results: dict[str, dict]) -> list[str]:
    """Create side-by-side evidence panels from the matrix results."""
    saved: list[str] = []
    out_dir = artifact_dir / "model_matrix"
    if 0 in denoise_results and 3 in denoise_results:
        saved.append(str(save_panel(
            out_dir / f"{denoise_results[0]['cid']}.jpg",
            out_dir / f"{denoise_results[3]['cid']}.jpg",
            "Real-CUGAN denoise_level 0 (no-denoise)",
            "Real-CUGAN denoise_level 3 (denoise3x)",
            out_dir / "compare_cugan_denoise_L0_vs_L3.png")))
    if "realesrgan-anime6b" in anime_results and "realesrgan-x4plus" in anime_results:
        saved.append(str(save_panel(
            out_dir / "anime_model.realesrgan_x4plus.png",
            out_dir / "anime_model.realesrgan_anime6b.png",
            "Real-ESRGAN x4plus (general)",
            "Real-ESRGAN anime6b (anime)",
            out_dir / "compare_anime6b_vs_x4plus.png", crop=(560, 320))))
    return saved


def section_models_sanity(chk: Checker, client: ApiClient, health: dict) -> None:
    chk.expect_eq("models.health.default",
                  (health.get("models") or {}).get("default"), "auto", "models")
    available = (health.get("models") or {}).get("available") or []
    chk.expect_eq("models.health.available", sorted(available), sorted(MODEL_IDS), "models")
    try:
        payload = client.models()
    except Exception as exc:  # noqa: BLE001
        chk.fail("models.api", f"{type(exc).__name__}: {exc}", "models")
        return
    models = payload.get("models") or []
    chk.expect_eq("models.api.count", len(models), len(MODEL_IDS), "models")
    chk.expect_eq("models.api.ids", [m.get("id") for m in models], MODEL_IDS, "models")
    for expected, actual in zip(MODELS_EXPECTED, models):
        cid = f"models.{expected['id']}"
        chk.expect_eq(f"{cid}.family", actual.get("family"), expected["family"], "models")
        chk.expect_eq(f"{cid}.available", actual.get("available"), True, "models")


def section_model_matrix(chk: Checker, client: ApiClient, artifact_dir: Path,
                         args) -> dict:
    """One real 720p->1080p job per model; returns {model: result}."""
    results: dict[str, dict] = {}
    for model_id in args.models:
        cid = f"matrix.{model_id.replace('-', '_')}"
        result = run_single_job(
            chk, client, artifact_dir, cid, MATRIX_SAMPLE, TARGET_720P, SIZE_1080P,
            "JPEG", "jpg", {"model": model_id, "format": "auto"},
            job_timeout=args.job_timeout)
        if result is None:
            continue
        results[model_id] = result
        chk.pass_(f"{cid}.perf", f"720p->1080p elapsed_sec={result['elapsed_sec']}s "
                                 f"wall={result['wall_sec']}s model_used={result['model_used']!r}")
        if result["size"] != list(SIZE_1080P):
            chk.fail(f"{cid}.size_again", f"unexpected {result['size']}", "matrix")
    return results


def section_model_4k(chk: Checker, client: ApiClient, artifact_dir: Path, args) -> dict:
    """Per-model 4K job (3840x2160); returns {model: result}."""
    results: dict[str, dict] = {}
    if args.no_4k:
        chk.skip("matrix_4k", "disabled by --no-4k", "perf")
        return results
    for model_id in args.models:
        cid = f"matrix4k.{model_id.replace('-', '_')}"
        # For auto, ratio 3.0 (> 2) must resolve to the x4 model.
        used_hints = ({"must": ["x4"], "must_not": ["anime"]}
                      if model_id == "auto" else None)
        result = run_single_job(
            chk, client, artifact_dir, cid, MATRIX_SAMPLE, TARGET_4K, SIZE_4K,
            "JPEG", "jpg", {"model": model_id, "format": "auto"},
            job_timeout=max(args.job_timeout, 240.0), used_hints=used_hints)
        if result is None:
            continue
        results[model_id] = result
        elapsed = result["elapsed_sec"]
        if isinstance(elapsed, (int, float)):
            ok = 0 < elapsed <= 180.0
            chk.expect(f"{cid}.within_180s", ok,
                       f"4K elapsed_sec={elapsed}s (contract <=180s)", "perf")
            if ok:
                chk.pass_(f"{cid}.perf", f"4K elapsed_sec={elapsed}s "
                                         f"wall={result['wall_sec']}s")
    return results


def section_realcugan_denoise(chk: Checker, client: ApiClient, artifact_dir: Path,
                              args) -> dict:
    """realcugan denoise_level 0..3 on the Gaussian-noise sample."""
    results: dict[int, dict] = {}
    for level in DENOISE_LEVELS:
        cid = f"denoise.cugan_level{level}"
        result = run_single_job(
            chk, client, artifact_dir, cid, NOISY_SAMPLE, TARGET_720P, SIZE_1080P,
            "JPEG", "jpg",
            {"model": "realcugan", "denoise_level": level, "format": "auto"},
            job_timeout=args.job_timeout, expected_denoise_level=level)
        if result is not None:
            results[level] = result
    if 0 in results and 3 in results:
        diff = mean_abs_diff(results[0]["image"], results[3]["image"])
        chk.expect("denoise.cugan_level0_vs_3_diff", diff > 0.5,
                   f"mean|level0-level3|={diff:.2f} levels (threshold >0.5)", "denoise")
        chk.pass_("denoise.cugan_level0_vs_3_metrics",
                  f"mean_abs_diff={diff:.2f}, rmse={rmse(results[0]['image'], results[3]['image']):.2f}, "
                  f"lapvar L0={laplacian_variance(results[0]['image']):.0f} "
                  f"L3={laplacian_variance(results[3]['image']):.0f}")
    for a, b in ((0, 1), (1, 2), (2, 3)):
        if a in results and b in results:
            diff = mean_abs_diff(results[a]["image"], results[b]["image"])
            chk.pass_(f"denoise.cugan_level{a}_vs_{b}",
                      f"mean_abs_diff={diff:.2f} levels")
    return results


def section_anime_model_diff(chk: Checker, client: ApiClient, artifact_dir: Path,
                             args) -> dict:
    """anime6b vs x4plus on the anime-style sample must differ measurably."""
    results: dict[str, dict] = {}
    for model_id in ("realesrgan-anime6b", "realesrgan-x4plus"):
        cid = f"anime_model.{model_id.replace('-', '_')}"
        result = run_single_job(
            chk, client, artifact_dir, cid, ANIME_SAMPLE, TARGET_720P, SIZE_1080P,
            "PNG", "png", {"model": model_id, "format": "auto"},
            job_timeout=args.job_timeout)
        if result is not None:
            results[model_id] = result
    if "realesrgan-anime6b" in results and "realesrgan-x4plus" in results:
        a = results["realesrgan-anime6b"]["image"]
        b = results["realesrgan-x4plus"]["image"]
        diff = mean_abs_diff(a, b)
        chk.expect("anime_model.outputs_differ", diff > 1.0,
                   f"mean|anime6b-x4plus|={diff:.2f} levels (threshold >1.0)", "anime")
        chk.pass_("anime_model.diff_metrics",
                  f"mean_abs_diff={diff:.2f}, rmse={rmse(a, b):.2f}, "
                  f"lapvar anime6b={laplacian_variance(a):.0f} "
                  f"x4plus={laplacian_variance(b):.0f}")
        if diff <= 3.0:
            chk.warn("anime_model.diff_strength",
                     f"mean_abs_diff={diff:.2f} is small; anime6b may not be visibly different",
                     "anime")
    return results


def section_rrdb_denoise(chk: Checker, client: ApiClient, artifact_dir: Path, args) -> dict:
    """rrdb denoise=false vs true on the Gaussian-noise sample."""
    results: dict[bool, dict] = {}
    for denoise in (False, True):
        cid = f"denoise.rrdb_{'on' if denoise else 'off'}"
        result = run_single_job(
            chk, client, artifact_dir, cid, NOISY_SAMPLE, TARGET_720P, SIZE_1080P,
            "JPEG", "jpg",
            {"model": "realesrgan-x4plus", "denoise": denoise, "format": "auto"},
            job_timeout=args.job_timeout)
        if result is not None:
            results[denoise] = result
    if False in results and True in results:
        off = results[False]["image"]
        on = results[True]["image"]
        diff = mean_abs_diff(off, on)
        chk.expect("denoise.rrdb_on_off_diff", diff > 0.3,
                   f"mean|denoise_on-denoise_off|={diff:.2f} levels (threshold >0.3)",
                   "denoise")
        lap_off = laplacian_variance(off)
        lap_on = laplacian_variance(on)
        chk.pass_("denoise.rrdb_on_off_metrics",
                  f"mean_abs_diff={diff:.2f}, rmse={rmse(off, on):.2f}, "
                  f"lapvar off={lap_off:.0f} on={lap_on:.0f}")
        if lap_on < lap_off:
            chk.pass_("denoise.rrdb_reduces_noise",
                      f"lapvar {lap_off:.0f} -> {lap_on:.0f} (noise energy reduced)")
        else:
            chk.warn("denoise.rrdb_reduces_noise",
                     f"lapvar did not decrease: off={lap_off:.0f} on={lap_on:.0f}", "denoise")
    return results


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PicPolish R2 model matrix")
    parser.add_argument("--base-url", default="http://127.0.0.1:8796")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--job-timeout", type=float, default=300.0)
    parser.add_argument("--models", default=",".join(MODEL_IDS),
                        help=f"comma-separated model ids (default: all {len(MODEL_IDS)})")
    parser.add_argument("--no-4k", action="store_true", help="skip the per-model 4K jobs")
    parser.add_argument("--sections", default="matrix,perf4k,denoise,anime,rrdb",
                        help="comma-separated subset of matrix,perf4k,denoise,anime,rrdb")
    args = parser.parse_args(argv)
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    wanted = {s.strip() for s in args.sections.split(",") if s.strip()}

    setup_console()
    artifact_dir = (args.artifact_dir or
                    Path(__file__).resolve().parent / "artifacts" / "latest" / "matrix").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"PicPolish model matrix  base_url={args.base_url}")
    print(f"models  : {', '.join(args.models)}")
    print(f"artifacts: {artifact_dir}")
    print("=" * 78)

    chk = Checker()
    client = ApiClient(args.base_url, timeout=max(180.0, args.job_timeout))
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started = time.monotonic()
    exit_code = 0
    model_results: dict[str, dict] = {}
    results_4k: dict[str, dict] = {}
    denoise_results: dict[int, dict] = {}
    anime_results: dict[str, dict] = {}
    rrdb_results: dict[bool, dict] = {}
    comparisons: list[str] = []
    health: dict = {}
    try:
        try:
            health = client.health()
        except Exception as exc:  # noqa: BLE001
            print(f"FATAL: {args.base_url} unreachable: {type(exc).__name__}: {exc}")
            return 2
        engine = health.get("engine") or {}
        print(f"engine: backend={engine.get('backend')!r} device={engine.get('device')!r} "
              f"model={engine.get('model')!r} gpu={engine.get('gpu')!r}")
        if "matrix" in wanted:
            try:
                section_models_sanity(chk, client, health)
            except Exception as exc:  # noqa: BLE001
                chk.fail("models.exception", f"{type(exc).__name__}: {exc}", "models")
                traceback.print_exc()
            try:
                model_results = section_model_matrix(chk, client, artifact_dir, args)
            except Exception as exc:  # noqa: BLE001
                chk.fail("matrix.exception", f"{type(exc).__name__}: {exc}", "matrix")
                traceback.print_exc()
        if "perf4k" in wanted:
            try:
                results_4k = section_model_4k(chk, client, artifact_dir, args)
            except Exception as exc:  # noqa: BLE001
                chk.fail("perf4k.exception", f"{type(exc).__name__}: {exc}", "perf")
                traceback.print_exc()
        if "denoise" in wanted:
            try:
                denoise_results = section_realcugan_denoise(chk, client, artifact_dir, args)
            except Exception as exc:  # noqa: BLE001
                chk.fail("denoise.exception", f"{type(exc).__name__}: {exc}", "denoise")
                traceback.print_exc()
        if "anime" in wanted:
            try:
                anime_results = section_anime_model_diff(chk, client, artifact_dir, args)
            except Exception as exc:  # noqa: BLE001
                chk.fail("anime.exception", f"{type(exc).__name__}: {exc}", "anime")
                traceback.print_exc()
        if "rrdb" in wanted:
            try:
                rrdb_results = section_rrdb_denoise(chk, client, artifact_dir, args)
            except Exception as exc:  # noqa: BLE001
                chk.fail("rrdb.exception", f"{type(exc).__name__}: {exc}", "denoise")
                traceback.print_exc()
        try:
            comparisons = build_comparisons(artifact_dir, denoise_results, anime_results)
            if comparisons:
                chk.pass_("matrix.comparisons", ", ".join(Path(p).name for p in comparisons))
        except Exception as exc:  # noqa: BLE001 - evidence only
            chk.warn("matrix.comparisons", f"{type(exc).__name__}: {exc}")
    finally:
        client.close()
        duration = time.monotonic() - started
        summary = chk.summary()

        def slim(result: dict | None) -> dict | None:
            if result is None:
                return None
            return {k: v for k, v in result.items() if k != "image"}

        perf = {
            model: {"720p_sec": model_results.get(model, {}).get("elapsed_sec"),
                    "720p_wall_sec": model_results.get(model, {}).get("wall_sec"),
                    "4k_sec": results_4k.get(model, {}).get("elapsed_sec"),
                    "4k_wall_sec": results_4k.get(model, {}).get("wall_sec")}
            for model in args.models
        }
        diffs = {
            "anime6b_vs_x4plus_mean_abs": (
                round(mean_abs_diff(anime_results["realesrgan-anime6b"]["image"],
                                    anime_results["realesrgan-x4plus"]["image"]), 3)
                if len(anime_results) == 2 else None),
            "rrdb_denoise_on_off_mean_abs": (
                round(mean_abs_diff(rrdb_results[False]["image"], rrdb_results[True]["image"]), 3)
                if len(rrdb_results) == 2 else None),
            "cugan_level0_vs_level3_mean_abs": (
                round(mean_abs_diff(denoise_results[0]["image"], denoise_results[3]["image"]), 3)
                if 0 in denoise_results and 3 in denoise_results else None),
        }
        payload = {
            "suite": "model-matrix",
            "base_url": args.base_url,
            "started_at": started_at,
            "duration_sec": round(duration, 3),
            "engine": health.get("engine"),
            "models_available": (health.get("models") or {}).get("available"),
            "models": [slim(r) for r in model_results.values()],
            "models_4k": [slim(r) for r in results_4k.values()],
            "denoise_levels": [slim(r) for r in denoise_results.values()],
            "anime_models": [slim(r) for r in anime_results.values()],
            "rrdb_denoise": [slim(r) for r in rrdb_results.values()],
            "perf": perf,
            "diffs": diffs,
            "comparisons": comparisons,
            "summary": summary,
            "checks": chk.to_json(),
            "artifacts": str(artifact_dir),
        }
        out = args.json_out or (artifact_dir / "model_matrix.json")
        write_json(out, payload)
        print("-" * 78)
        print("PERF TABLE (elapsed_sec / wall_sec):")
        for model, row in perf.items():
            print(f"  {model:26s} 720p->1080p: {row['720p_sec']!s:>8s} / "
                  f"{row['720p_wall_sec']!s:>8s}   4K: {row['4k_sec']!s:>8s} / "
                  f"{row['4k_wall_sec']!s:>8s}")
        print(f"DIFFS: {diffs}")
        print(f"JSON summary: {out}")
        print(f"MATRIX SUMMARY: PASS={summary['pass']} FAIL={summary['fail']} "
              f"WARN={summary['warn']} SKIP={summary['skip']} elapsed={duration:.1f}s")
        if summary["failed_ids"]:
            print("FAILED CHECKS: " + ", ".join(summary["failed_ids"]))
        exit_code = 0 if summary["fail"] == 0 else 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
