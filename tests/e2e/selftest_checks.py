#!/usr/bin/env python
"""Offline self-test for the PicPolish E2E assertion helpers.

Proves that the checks themselves are sound before they are trusted to judge
the application:

1. the hard-coded plan table matches ``app/target.py`` (same as
   ``e2e_windows.py --self-test-plan``);
2. the content-validity gate really catches a broken tiling pass: we take a
   clean gradient, simulate (a) large black blocks, (b) a single black tile,
   (c) a thin black seam -- the gate must flag each, and must NOT flag the clean
   image nor the real GPU output from the latest acceptance run (if present);
3. alpha handling: the RGBA sample is composited onto neutral grey before the
   black-ratio test, so its transparent hole is not mistaken for black content;
4. Content-Disposition filename parsing (plain + RFC 5987);
5. (R2) the model registry matches the documented 6 models / order / family,
   noise + denoise image metrics behave as expected, and the ``model_used``
   matcher accepts the right families while rejecting wrong ones.

Run::

    .venv\\Scripts\\python.exe tests\\e2e\\selftest_checks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import (  # noqa: E402
    MODEL_IDS,
    MODELS_EXPECTED,
    PROJECT_ROOT,
    SAMPLES_DIR,
    content_disposition_filename,
    content_problems,
    differing_pixel_ratio,
    image_info,
    image_to_bytes,
    laplacian_variance,
    mean_abs_diff,
    model_used_matches,
    parse_content_disposition,
    rmse,
    setup_console,
    tile_metrics,
)
from e2e_windows import (  # noqa: E402
    PLAN_CASES,
    SEAM_TOL,
    TILE_MEAN_TOL,
    self_test_plan,
)

FAILURES: list[str] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def detect_problems(im: Image.Image) -> tuple[list[str], dict]:
    info = image_info(image_to_bytes(im))
    problems = content_problems(info)
    metrics = tile_metrics(im, grid=4, block=32)
    tile_bad = (
        metrics["max_cell_deviation"] >= TILE_MEAN_TOL
        or metrics["max_block_deviation"] >= TILE_MEAN_TOL
        or metrics["column_jump_max"] >= SEAM_TOL
        or metrics["row_jump_max"] >= SEAM_TOL
    )
    return problems, metrics, tile_bad


def main() -> int:
    setup_console()
    print("=" * 78)
    print("PicPolish E2E self-test (assertion logic only, no server needed)")
    print("=" * 78)

    # 1) expected plan table vs app/target.py (delegates to e2e_windows)
    code = self_test_plan()
    record("plan.expected_table_matches_app_target", code == 0,
           f"{len(PLAN_CASES)} cases")

    # 2) broken-tiling detection
    gradient_path = SAMPLES_DIR / "solid_gradient_700x700.png"
    if not gradient_path.is_file():
        record("samples.gradient_present", False, f"missing {gradient_path}")
        return 1
    with Image.open(gradient_path) as src:
        good = (src.convert("RGB").resize((1080, 1080), Image.LANCZOS)
                .filter(ImageFilter.UnsharpMask(radius=1.2, percent=45, threshold=3)))

    problems, metrics, tile_bad = detect_problems(good)
    record("detect.clean_gradient_not_flagged", not problems and not tile_bad,
           f"cell_dev={metrics['max_cell_deviation']:.2f} "
           f"block_dev={metrics['max_block_deviation']:.2f} "
           f"col_jump={metrics['column_jump_max']:.2f}")

    broken = good.copy()
    pixels = broken.load()
    for x in range(int(1080 * 0.4), 1080):
        for y in range(int(1080 * 0.92)):
            pixels[x, y] = (0, 0, 0)
    problems, metrics, tile_bad = detect_problems(broken)
    ratio = image_info(image_to_bytes(broken))["black_ratio"]
    record("detect.black_blocks_flagged", bool(problems) and tile_bad,
           f"black_ratio={ratio:.1%}, problems={problems}, "
           f"cell_dev={metrics['max_cell_deviation']:.1f}")

    one_tile = good.copy()
    pixels = one_tile.load()
    for x in range(256, 616):
        for y in range(256, 616):
            pixels[x, y] = (0, 0, 0)
    problems, _metrics, tile_bad = detect_problems(one_tile)
    record("detect.single_black_tile_flagged", bool(problems) or tile_bad,
           f"problems={problems}, tile_bad={tile_bad}")

    seam = good.copy()
    pixels = seam.load()
    for x in range(540, 542):
        for y in range(1080):
            pixels[x, y] = (0, 0, 0)
    problems, metrics, tile_bad = detect_problems(seam)
    record("detect.thin_black_seam_flagged", (not problems) and tile_bad,
           f"black_ratio below gate but column_jump={metrics['column_jump_max']:.1f}")

    # Real GPU output from the latest acceptance run (optional).
    gpu_output = HERE / "artifacts" / "latest" / "functional" / "tile" / "gradient_1080p.png"
    if gpu_output.is_file():
        with Image.open(gpu_output) as gpu:
            problems, metrics, tile_bad = detect_problems(gpu.convert("RGB"))
        record("detect.real_gpu_output_not_flagged", not problems and not tile_bad,
               f"cell_dev={metrics['max_cell_deviation']:.2f} "
               f"block_dev={metrics['max_block_deviation']:.2f} "
               f"col_jump={metrics['column_jump_max']:.2f}")
    else:
        print("[SKIP] detect.real_gpu_output_not_flagged  --  no artifact from a GPU run yet")

    # 3) dark-source handling: legitimately dark photos must be judged against
    #    their own input, while extra black blocks must still be caught.
    dark = Image.new("RGB", (600, 400), (26, 26, 26))
    dark_pixels = dark.load()
    for x in range(0, 132):          # ~22 % naturally black columns
        for y in range(400):
            dark_pixels[x, y] = (0, 0, 0)
    dark_ref = image_info(image_to_bytes(dark))
    problems = content_problems(image_info(image_to_bytes(dark)), reference=dark_ref)
    record("detect.dark_photo_accepted_with_reference", not problems,
           f"black={dark_ref['black_ratio']:.1%}, mean={dark_ref['mean']:.1f}")

    dark_broken = dark.copy()
    broken_pixels = dark_broken.load()
    for x in range(250, 600):        # extra 58 % black -> broken tiler
        for y in range(400):
            broken_pixels[x, y] = (0, 0, 0)
    problems = content_problems(image_info(image_to_bytes(dark_broken)), reference=dark_ref)
    record("detect.dark_photo_broken_flagged", bool(problems), "; ".join(problems))

    # 1b) model registry expectations vs app/models.py (R2; optional until it lands)
    models_py = PROJECT_ROOT / "app" / "models.py"
    if models_py.is_file():
        try:
            from app import models as models_module  # type: ignore

            payload = models_module.list_models()
            ids = [m.get("id") for m in (payload.get("models") or [])]
            record("models.registry_order", ids == MODEL_IDS, f"ids={ids}")
            registry = getattr(models_module, "MODEL_REGISTRY", {})
            for expected in MODELS_EXPECTED:
                entry = registry.get(expected["id"])
                record(f"models.registry.{expected['id']}.family",
                       isinstance(entry, dict) and entry.get("family") == expected["family"],
                       f"family={entry.get('family') if isinstance(entry, dict) else entry!r}")
            try:
                available = list(models_module.available_model_ids())
            except Exception as exc:  # noqa: BLE001
                record("models.weights_available", False, f"{type(exc).__name__}: {exc}")
            else:
                record("models.weights_available", sorted(available) == sorted(MODEL_IDS),
                       f"available={sorted(available)}")
        except Exception as exc:  # noqa: BLE001
            record("models.registry_importable", False, f"{type(exc).__name__}: {exc}")
    else:
        print("[SKIP] models.registry  --  app/models.py not present yet")

    # 4b) denoise / difference metrics must behave as intended.
    clean_path = SAMPLES_DIR / "photo_1280x720.jpg"
    noisy_path = SAMPLES_DIR / "noisy_gauss_1280x720.jpg"
    if clean_path.is_file() and noisy_path.is_file():
        with Image.open(clean_path) as clean_im:
            clean_im.load()
            clean_img = clean_im.copy()
        with Image.open(noisy_path) as noisy_im:
            noisy_im.load()
            noisy_img = noisy_im.copy()
        clean_lap = laplacian_variance(clean_img)
        noisy_lap = laplacian_variance(noisy_img)
        record("noise.noisy_sample_has_more_hf_energy", noisy_lap > clean_lap * 1.2,
               f"lapvar clean={clean_lap:.0f} noisy={noisy_lap:.0f}")
        diff = mean_abs_diff(clean_img, noisy_img)
        record("diff.clean_vs_noisy_detected", diff > 2.0,
               f"mean_abs_diff={diff:.2f}, rmse={rmse(clean_img, noisy_img):.2f}, "
               f"pixels_differ={differing_pixel_ratio(clean_img, noisy_img):.1%}")
        record("diff.identical_is_zero", mean_abs_diff(clean_img, clean_img) == 0.0)
        # A Gaussian blur stands in for "denoise": high-frequency energy must drop.
        blurred = noisy_img.filter(ImageFilter.GaussianBlur(1.2))
        record("denoise.blur_reduces_hf_energy",
               laplacian_variance(blurred) < noisy_lap,
               f"lapvar noisy={noisy_lap:.0f} blurred={laplacian_variance(blurred):.0f}")
        record("denoise.blur_change_detected", mean_abs_diff(noisy_img, blurred) > 0.5,
               f"mean_abs_diff={mean_abs_diff(noisy_img, blurred):.2f}")
    else:
        record("noise.samples_present", False,
               f"missing {clean_path} or {noisy_path}")

    # 4c) model_used matching rules (tolerant but must reject the wrong family).
    cases = [
        ("auto", "RealESRGAN_x2plus", True),
        ("realesrgan-x4plus", "RealESRGAN_x4plus", True),
        ("realesrgan-anime6b", "RealESRGAN_x4plus_anime_6B", True),
        ("realcugan", "Real-CUGAN 2x denoise1x", True),
        ("realcugan-conservative", "Real-CUGAN 2x conservative", True),
        ("realcugan", "RealESRGAN_x4plus", False),
        ("realesrgan-x4plus", "RealESRGAN_x4plus_anime_6B", False),
        ("auto", "", False),
    ]
    for model_id, used, expected_ok in cases:
        ok, detail = model_used_matches(model_id, used)
        record(f"model_used.{model_id}.{used or 'empty'}",
               ok == expected_ok, detail)

    # 4) alpha composite handling
    alpha_path = SAMPLES_DIR / "alpha_800x600.png"
    if alpha_path.is_file():
        info = image_info(alpha_path.read_bytes())
        record("alpha.transparent_hole_not_black", info["black_ratio"] < 0.005,
               f"black_ratio={info['black_ratio']:.2%}, "
               f"transparent={info['alpha_transparent_ratio']:.1%}, "
               f"alpha_min={info['alpha_min']:.0f}")
        record("alpha.channel_detected", info["alpha_min"] < 32 and info["alpha_max"] > 240,
               f"alpha range [{info['alpha_min']:.0f}, {info['alpha_max']:.0f}]")
    else:
        record("alpha.sample_present", False, f"missing {alpha_path}")

    # 5) Content-Disposition parsing
    plain = 'attachment; filename="photo_1280x720_1080p.jpg"'
    record("disposition.plain", content_disposition_filename(plain) == "photo_1280x720_1080p.jpg",
           plain)
    rfc = "attachment; filename*=UTF-8''%E7%85%A7%E7%89%87_1080p.png"
    record("disposition.rfc5987", content_disposition_filename(rfc) == "照片_1080p.png",
           f"parsed={content_disposition_filename(rfc)!r}")
    record("disposition.type", parse_content_disposition(plain).get("_type") == "attachment")

    print("=" * 78)
    if FAILURES:
        print(f"SELF-TEST FAILED: {len(FAILURES)} -> {', '.join(FAILURES)}")
        return 1
    print("SELF-TEST: OK (all assertion helpers behave as expected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
