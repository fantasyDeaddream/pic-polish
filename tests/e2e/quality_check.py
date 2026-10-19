#!/usr/bin/env python
"""Quality + performance spot-check for PicPolish real-photo results.

For each sample we submit one job at a fixed tier, download the produced image
and compare it against a **Lanczos upscale of the same input to the very same
output size** -- the "平滑真实" baseline the product promises to beat:

* content    -- hard gate: black pixels (RGB max < 8) < 0.5 % and global mean
  > 10, so a broken tiler cannot pass on size/format alone;
* sharpness  -- variance of the 4-neighbour Laplacian response; the SR result
  must not be degraded vs Lanczos (>= 0.4x, per Lead guidance, because the JPEG
  result carries codec noise) and > 1.0x is reported as "clearly sharper";
* integrity  -- never all-white / flat (mean, std, saturated-pixel ratios);
* colour     -- per-channel mean must stay close to the Lanczos baseline (no
  severe colour cast);
* performance-- the 1280x720 -> 1920x1080 GPU job must complete within 45 s
  (docs/CONTRACT.md section 5).

A side-by-side ``Lanczos | result`` crop is written per case for eyeballing.

Exit code 0 iff every blocking check passed.  Like ``e2e_windows.py`` this
script never starts/stops the server.

Usage::

    .venv\\Scripts\\python.exe tests\\e2e\\quality_check.py \\
        --base-url http://127.0.0.1:8796 --artifact-dir tests\\e2e\\artifacts\\latest
"""
from __future__ import annotations

import argparse
import io
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    SAMPLES_DIR,
    ApiClient,
    Checker,
    bicubic_baseline,
    content_problems,
    image_stats,
    laplacian_variance,
    load_samples,
    resample_baseline,
    setup_console,
    write_json,
)

TARGET = "1080p"
MAX_GPU_SECONDS = 45.0  # docs/CONTRACT.md section 5: 720P -> 1080p on GPU

# sample -> expected output after /api/plan for TARGET; perf=True marks the
# 720p milestone measured against the 45 s target.
CASES = [
    {"sample": "photo_1280x720.jpg", "out": (1920, 1080), "label": "landscape",
     "perf": True, "crop": (480, 270)},
    {"sample": "portrait_720x1280.jpg", "out": (1080, 1920), "label": "portrait",
     "perf": False, "crop": (270, 480)},
]

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
]


def _font(size: int):
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _channel_delta(a: Image.Image, b: Image.Image) -> list[float]:
    arr_a = np.asarray(a.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(axis=0)
    arr_b = np.asarray(b.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(axis=0)
    return [abs(float(x) - float(y)) for x, y in zip(arr_a, arr_b)]


def _cast_spread(im: Image.Image) -> float:
    means = np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(axis=0)
    return float(means.max() - means.min())


def _save_comparison(result: Image.Image, baseline: Image.Image, case: dict,
                     out_path: Path) -> None:
    """Center-crop bicubic vs SR side by side, labelled, for eyeball review."""
    crop_w, crop_h = case["crop"]
    def crop(im: Image.Image) -> Image.Image:
        x = max(0, (im.width - crop_w) // 2)
        y = max(0, (im.height - crop_h) // 2)
        return im.crop((x, y, x + crop_w, y + crop_h))
    left = crop(baseline)
    right = crop(result)
    font = _font(22)
    panel = Image.new("RGB", (crop_w * 2 + 12, crop_h + 34), (24, 24, 24))
    panel.paste(left.convert("RGB"), (0, 34))
    panel.paste(right.convert("RGB"), (crop_w + 12, 34))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 6), "Lanczos baseline", font=font, fill=(230, 230, 230))
    draw.text((crop_w + 18, 6), "PicPolish result", font=font, fill=(120, 255, 160))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path, format="PNG", optimize=True)


def run_case(chk: Checker, client: ApiClient, case: dict, artifact_dir: Path,
             backend: str, args) -> dict:
    name = case["sample"]
    paths = load_samples([name])
    label = case["label"]
    cid = f"quality.{label}"
    record: dict = {"sample": name, "label": label}

    # Expected output size must come from the server's own plan endpoint.
    with Image.open(paths[0]) as source_probe:
        input_w, input_h = source_probe.size
    plan_resp = client.plan([{"name": name, "width": input_w, "height": input_h}], TARGET)
    if plan_resp.status_code == 200:
        images = plan_resp.json().get("images") or []
        planned = tuple((images[0].get("output") or {}).get(k) for k in ("width", "height")) \
            if images else (None, None)
        chk.expect_eq(f"{cid}.plan.output", planned, case["out"], "quality")
    else:
        chk.fail(f"{cid}.plan.http_status", f"{plan_resp.status_code}: {plan_resp.text[:200]}",
                 "quality")

    started = time.monotonic()
    try:
        resp = client.create_job(paths, target=TARGET, format="auto")
        if not chk.expect_eq(f"{cid}.job.create_status", resp.status_code, 202, "quality"):
            return record
        job_id = resp.json().get("job_id")
        final, _poll_elapsed, _samples = client.wait_job(job_id, timeout=args.job_timeout)
        wall = time.monotonic() - started
        record.update({"job_id": job_id, "status": final.get("status"),
                       "wall_sec": round(wall, 3),
                       "elapsed_sec": final.get("elapsed_sec")})
        if not chk.expect_eq(f"{cid}.job.terminal_status", final.get("status"), "done",
                             "quality"):
            return record
        file_entry = (final.get("files") or [{}])[0]
        chk.expect_eq(f"{cid}.job.output_size", file_entry.get("output"),
                      {"width": case["out"][0], "height": case["out"][1]}, "quality")

        download = client.file_bytes(job_id, 0, download=True)
        if not chk.expect_eq(f"{cid}.download.status", download.status_code, 200, "quality"):
            return record
    finally:
        try:
            if "job_id" in record:
                client.delete_job(record["job_id"])
        except Exception:  # noqa: BLE001
            pass

    # ---- metrics ------------------------------------------------------
    with Image.open(io.BytesIO(download.content)) as produced:
        produced.load()
        assert produced.size == case["out"], f"{name}: {produced.size} != {case['out']}"
        with Image.open(paths[0]) as source:
            source.load()
            # Lead guidance: compare against the plain Lanczos baseline of the
            # identical input/size; bicubic is reported next to it for context.
            baseline = resample_baseline(source, case["out"], Image.LANCZOS)
            bicubic = bicubic_baseline(source, case["out"])
        lap_sr = laplacian_variance(produced)
        lap_lanczos = laplacian_variance(baseline)
        lap_bicubic = laplacian_variance(bicubic)
        stats = image_stats(produced)
        base_stats = image_stats(baseline)
        channel_delta = _channel_delta(produced, baseline)
        cast_spread = _cast_spread(produced)
        ratio = lap_sr / lap_lanczos if lap_lanczos else float("inf")
        bicubic_ratio = lap_sr / lap_bicubic if lap_bicubic else float("inf")

        record.update({
            "output": {"width": produced.width, "height": produced.height,
                       "format": produced.format, "mode": produced.mode},
            "sharpness": {"laplacian_sr": round(lap_sr, 2),
                          "laplacian_lanczos": round(lap_lanczos, 2),
                          "laplacian_bicubic": round(lap_bicubic, 2),
                          "ratio_vs_lanczos": round(ratio, 3),
                          "ratio_vs_bicubic": round(bicubic_ratio, 3)},
            "stats": {"mean": round(stats["mean"], 2), "std": round(stats["std"], 2),
                      "black_ratio": round(stats["black_ratio"], 5),
                      "white_ratio": round(stats["white_ratio"], 5),
                      "channel_means": [round(x, 2) for x in stats["channel_means"]],
                      "cast_spread": round(cast_spread, 2)},
            "baseline_stats": {"mean": round(base_stats["mean"], 2),
                               "std": round(base_stats["std"], 2),
                               "channel_means": [round(x, 2) for x in base_stats["channel_means"]]},
            "channel_delta_vs_lanczos": [round(x, 2) for x in channel_delta],
        })

        chk.expect(f"{cid}.output_format",
                   (produced.format or "").upper() in ("JPEG", "PNG"),
                   f"format={produced.format!r} mode={produced.mode!r}", "quality")

        # 1) content validity (hard): broken tiling -> black tiles / flat image.
        #    The Lanczos baseline doubles as the "natural darkness" reference so
        #    legitimately dark photos are judged against their own input.
        bad = content_problems(stats, reference=base_stats)
        chk.expect(f"{cid}.content_valid", not bad, "; ".join(bad), "quality")
        if stats["std"] < 5.0:
            chk.warn(f"{cid}.content_texture", f"std={stats['std']:.2f} (very flat output)",
                     "quality")

        # 2) sharpness vs the identical-size Lanczos baseline.  The result is
        #    JPEG-encoded so the raw variance comparison carries codec noise;
        #    per Lead guidance only a >=0.4x ratio is a hard requirement, while
        #    >1.0x counts as "clearly sharper".
        chk.expect(f"{cid}.sharpness_not_degraded", ratio > 0.4,
                   f"Laplacian var SR={lap_sr:.1f} vs Lanczos={lap_lanczos:.1f} "
                   f"(ratio {ratio:.2f}x, floor 0.40x; bicubic ratio {bicubic_ratio:.2f}x)",
                   "quality")
        if ratio > 1.0:
            chk.pass_(f"{cid}.sharper_than_lanczos",
                      f"{ratio:.2f}x sharper than Lanczos (SR={lap_sr:.1f}, "
                      f"Lanczos={lap_lanczos:.1f}, bicubic={lap_bicubic:.1f})", "quality")
        else:
            chk.warn(f"{cid}.sharper_than_lanczos",
                     f"not clearly sharper: SR={lap_sr:.1f}, Lanczos={lap_lanczos:.1f} "
                     f"(ratio {ratio:.2f}x) -- recommend a manual eye check", "quality")

        # 3) no severe colour cast relative to the Lanczos baseline
        worst_delta = max(channel_delta)
        chk.expect(f"{cid}.colour_consistent", worst_delta < 20.0,
                   f"per-channel |mean(SR)-mean(baseline)|={[round(x, 1) for x in channel_delta]}",
                   "quality")
        if cast_spread > max(80.0, _cast_spread(baseline) + 30.0):
            chk.warn(f"{cid}.colour_cast_spread",
                     f"channel mean spread {cast_spread:.1f} (baseline "
                     f"{_cast_spread(baseline):.1f})", "quality")
        else:
            chk.pass_(f"{cid}.colour_cast_spread", f"spread {cast_spread:.1f}", "quality")

        # Side-by-side crop for manual inspection.
        comparison = artifact_dir / f"{Path(name).stem}_lanczos_vs_result.png"
        try:
            _save_comparison(produced, baseline, case, comparison)
            record["comparison"] = str(comparison)
            chk.pass_(f"{cid}.comparison_saved", str(comparison), "quality")
        except Exception as exc:  # noqa: BLE001
            chk.warn(f"{cid}.comparison_saved", f"{type(exc).__name__}: {exc}", "quality")

    # ---- performance --------------------------------------------------
    if case["perf"]:
        elapsed = record.get("elapsed_sec")
        wall_value = record.get("wall_sec")
        measured = elapsed if isinstance(elapsed, (int, float)) else wall_value
        if backend == "realesrgan":
            chk.expect(f"{cid}.gpu_time_within_{int(MAX_GPU_SECONDS)}s",
                       isinstance(measured, (int, float)) and 0 < measured <= MAX_GPU_SECONDS,
                       f"720p->1080p took {measured!r}s (job.elapsed_sec={elapsed!r}, "
                       f"wall={wall_value!r}s, target <={MAX_GPU_SECONDS:.0f}s)", "perf")
        else:
            chk.warn(f"{cid}.gpu_time_within_{int(MAX_GPU_SECONDS)}s",
                     f"engine backend is {backend!r}, not realesrgan -- GPU timing not applicable "
                     f"(measured {measured!r}s)", "perf")
        chk.pass_(f"{cid}.perf_measured",
                  f"720p->1080p elapsed_sec={elapsed!r}s wall={wall_value!r}s backend={backend}",
                  "perf")
        record["perf"] = {"elapsed_sec": elapsed, "wall_sec": wall_value,
                          "target_sec": MAX_GPU_SECONDS, "backend": backend}
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PicPolish quality/performance check")
    parser.add_argument("--base-url", default="http://127.0.0.1:8796")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--job-timeout", type=float, default=180.0)
    args = parser.parse_args(argv)

    setup_console()
    artifact_dir = (args.artifact_dir or Path(tempfile.mkdtemp(prefix="picpolish_quality_"))).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"PicPolish quality/performance check  base_url={args.base_url}")
    print(f"artifacts: {artifact_dir}")
    print("=" * 78)

    chk = Checker()
    client = ApiClient(args.base_url, timeout=180.0)
    records: list[dict] = []
    health: dict = {}
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started = time.monotonic()
    exit_code = 0
    try:
        try:
            health = client.health()
        except Exception as exc:  # noqa: BLE001
            print(f"FATAL: {args.base_url} unreachable: {type(exc).__name__}: {exc}")
            return 2
        engine = health.get("engine") or {}
        backend = engine.get("backend")
        print(f"engine: backend={backend!r} device={engine.get('device')!r} "
              f"gpu={engine.get('gpu')!r} model={engine.get('model')!r}")
        chk.expect("quality.engine.available",
                   backend in ("realesrgan", "pil"), f"engine={engine!r}", "quality")
        for case in CASES:
            try:
                records.append(run_case(chk, client, case, artifact_dir, backend or "unknown",
                                        args))
            except Exception as exc:  # noqa: BLE001
                chk.fail(f"quality.{case['label']}.exception",
                         f"{type(exc).__name__}: {exc}", "quality")
                traceback.print_exc()
    finally:
        client.close()
        duration = time.monotonic() - started
        summary = chk.summary()
        payload = {
            "suite": "quality-check",
            "base_url": args.base_url,
            "started_at": started_at,
            "duration_sec": round(duration, 3),
            "engine": health.get("engine"),
            "cases": records,
            "summary": summary,
            "checks": chk.to_json(),
            "artifacts": str(artifact_dir),
        }
        out = args.json_out or (artifact_dir / "quality_check.json")
        write_json(out, payload)
        print("-" * 78)
        print(f"JSON summary: {out}")
        print(f"QUALITY SUMMARY: PASS={summary['pass']} FAIL={summary['fail']} "
              f"WARN={summary['warn']} elapsed={duration:.1f}s")
        if summary["failed_ids"]:
            print("FAILED CHECKS: " + ", ".join(summary["failed_ids"]))
        exit_code = 0 if summary["fail"] == 0 else 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
