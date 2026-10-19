"""Generate the deterministic PicPolish E2E sample set into ``samples/``.

The script is intentionally re-runnable: every file is rebuilt with fixed
parameters, and the resulting set stays small (well under the 60 MB budget).

Sources
-------
* Four real photographs are copied from ``C:\\Windows\\Web\\Wallpaper\\Theme1``
  (Windows 10/11 ships them, so they are always available on this machine).
* Derived images (720p / portrait / tiny / 4K / odd-ratio) are cropped and
  resized from those same photographs so the super-resolution quality check has
  real texture to work with.
* Two purely synthetic images cover cases no photo does: an RGBA image with a
  real alpha channel and a text/gradient chart meant for eyeball comparison.

Run with the project venv::

    .venv\\Scripts\\python.exe tests\\e2e\\make_samples.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
WALLPAPER_DIR = Path(r"C:\Windows\Web\Wallpaper\Theme1")

JPEG_QUALITY = 95
TOTAL_BUDGET_MB = 60

# Windows font fallbacks for the text/gradient chart (first existing wins).
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\seguisb.ttf"),
    Path(r"C:\Windows\Fonts\arialbd.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
]

# name -> usage note shown in the manifest and e2e output.
REAL_PHOTOS = {
    "photo_beach_1920x1200.jpg": "img1.jpg",   # sea stacks, runner, clouds
    "photo_cliff_1920x1200.jpg": "img3.jpg",   # granite cliff climber, fine texture
    "photo_night_1920x1200.jpg": "img4.jpg",   # milky-way tent, dark tones
    "photo_panorama_3840x1200.jpg": "img13.jpg",  # 4K-wide panorama, downscale input
}


def _open_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def _fit(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Cover-crop to exactly ``size`` (no distortion, no letterboxing)."""
    return ImageOps.fit(im, size, method=Image.LANCZOS, centering=(0.5, 0.5))


def _save_jpeg(im: Image.Image, path: Path, quality: int = JPEG_QUALITY) -> None:
    im.convert("RGB").save(path, format="JPEG", quality=quality, optimize=True,
                           progressive=False, subsampling=0)


def _save_png(im: Image.Image, path: Path) -> None:
    im.save(path, format="PNG", optimize=True)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def make_real_photos(manifest: list[dict]) -> dict[str, Image.Image]:
    photos: dict[str, Image.Image] = {}
    if not WALLPAPER_DIR.is_dir():
        raise SystemExit(f"素材源目录不存在：{WALLPAPER_DIR}")
    for out_name, src_name in REAL_PHOTOS.items():
        src = WALLPAPER_DIR / src_name
        if not src.is_file():
            raise SystemExit(f"素材源文件缺失：{src}")
        im = _open_rgb(src)
        dst = SAMPLES_DIR / out_name
        _save_jpeg(im, dst)
        photos[out_name] = im
        manifest.append({
            "name": out_name, "kind": "real-photo", "source": str(src),
            "width": im.width, "height": im.height,
            "note": f"从 {src_name} 复制的真实照片",
        })
        print(f"  [photo ] {out_name:38s} {im.width}x{im.height}")
    return photos


def make_derived(photos: dict[str, Image.Image], manifest: list[dict]) -> None:
    beach = photos["photo_beach_1920x1200.jpg"]
    cliff = photos["photo_cliff_1920x1200.jpg"]
    night = photos["photo_night_1920x1200.jpg"]
    pano = photos["photo_panorama_3840x1200.jpg"]

    # 720p landscape: the primary 720p -> 1080p GPU timing / quality input.
    im = _fit(cliff, (1280, 720))
    _save_jpeg(im, SAMPLES_DIR / "photo_1280x720.jpg")
    manifest.append({"name": "photo_1280x720.jpg", "kind": "derived",
                     "source": "photo_cliff_1920x1200.jpg", "width": 1280, "height": 720,
                     "note": "真实照片裁切缩放；720p→1080p 性能/质量主样本"})
    print(f"  [derive] photo_1280x720.jpg                 1280x720")

    # Portrait (9:16) from the beach photo.
    im = _fit(beach, (720, 1280))
    _save_jpeg(im, SAMPLES_DIR / "portrait_720x1280.jpg")
    manifest.append({"name": "portrait_720x1280.jpg", "kind": "derived",
                     "source": "photo_beach_1920x1200.jpg", "width": 720, "height": 1280,
                     "note": "竖图；720p@短边 → 720x1280"})
    print(f"  [derive] portrait_720x1280.jpg              720x1280")

    # Tiny low-resolution input (upscale stress / minimum-size handling).
    im = _fit(cliff, (320, 240))
    _save_png(im, SAMPLES_DIR / "tiny_320x240.png")
    manifest.append({"name": "tiny_320x240.png", "kind": "derived",
                     "source": "photo_cliff_1920x1200.jpg", "width": 320, "height": 240,
                     "note": "低分辨率 PNG，测小图放大"})
    print(f"  [derive] tiny_320x240.png                   320x240")

    # 4K UHD input: only reachable by up-scaling the 3840x1200 panorama first.
    wide = pano.resize((pano.width * 2160 // pano.height, 2160), Image.LANCZOS)
    im = ImageOps.fit(wide, (3840, 2160), method=Image.LANCZOS)
    _save_jpeg(im, SAMPLES_DIR / "large_3840x2160.jpg", quality=92)
    manifest.append({"name": "large_3840x2160.jpg", "kind": "derived",
                     "source": "photo_panorama_3840x1200.jpg", "width": 3840, "height": 2160,
                     "note": "4K 大图；1080p 档位下为 downscale"})
    print(f"  [derive] large_3840x2160.jpg                3840x2160")

    # Odd ratio 1000x667 -> 1080p must snap the free side to 1616 px.
    im = _fit(night, (1000, 667))
    _save_jpeg(im, SAMPLES_DIR / "odd_1000x667.jpg")
    manifest.append({"name": "odd_1000x667.jpg", "kind": "derived",
                     "source": "photo_night_1920x1200.jpg", "width": 1000, "height": 667,
                     "note": "1000x667；1080p 输出 1616x1080（8px 吸附）"})
    print(f"  [derive] odd_1000x667.jpg                   1000x667")


def make_alpha(manifest: list[dict]) -> None:
    """RGBA sample with a genuine alpha channel (opaque art on transparent bg)."""
    w, h = 800, 600
    # Opaque diagonal gradient panel built with numpy (fast, deterministic).
    t = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    u = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(40 + 180 * t + 30 * u, 0, 255)
    rgba[..., 1] = np.clip(200 - 120 * t + 20 * u, 0, 255)
    rgba[..., 2] = np.clip(120 + 100 * (1 - t) - 40 * u, 0, 255)
    rgba[..., 3] = 255
    im = Image.fromarray(rgba, mode="RGBA")
    draw = ImageDraw.Draw(im)

    # Fully transparent circular hole (alpha exactly 0); neutral RGB underneath
    # so a later composite onto grey is unaffected.
    draw.ellipse([w // 2 - 120, h // 2 - 120, w // 2 + 120, h // 2 + 120],
                 fill=(128, 128, 128, 0))
    # Semi-transparent ring around it.
    draw.ellipse([w // 2 - 160, h // 2 - 160, w // 2 + 160, h // 2 + 160],
                 outline=(255, 255, 255, 140), width=18)
    # Opaque text and corner marks, so sharp edges survive the SR round-trip.
    draw.text((40, 40), "PicPolish ALPHA TEST", font=_font(46), fill=(255, 255, 255, 255))
    draw.text((40, h - 90), "RGBA 800x600 / transparent center", font=_font(26),
              fill=(20, 20, 20, 230))
    for corner in ((30, 30), (w - 30, 30), (30, h - 30), (w - 30, h - 30)):
        draw.ellipse([corner[0] - 12, corner[1] - 12, corner[0] + 12, corner[1] + 12],
                     fill=(255, 220, 0, 255))

    path = SAMPLES_DIR / "alpha_800x600.png"
    _save_png(im, path)
    manifest.append({"name": "alpha_800x600.png", "kind": "synthetic",
                     "source": "generated", "width": w, "height": h,
                     "note": "RGBA 透明通道；中心 alpha=0 圆洞；测 alpha 通道单独缩放"})
    print(f"  [synth ] alpha_800x600.png                  800x600 RGBA")


def make_text_gradient(manifest: list[dict]) -> None:
    """Smooth gradient + crisp text/fine lines: subjective sharpness chart."""
    w, h = 1280, 720
    # Smooth gradient via numpy; deterministic and quick.
    t = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    u = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(30 + 200 * t + 15 * u, 0, 255)
    rgb[..., 1] = np.clip(10 + 120 * u + 60 * t, 0, 255)
    rgb[..., 2] = np.clip(120 + 100 * (1 - t) - 40 * u, 0, 255)
    im = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(im)
    # Fine 1 px grid: high-frequency detail that a good upscaler must preserve.
    for x in range(0, w, 16):
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 60), width=1)
    for y in range(0, h, 16):
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 40), width=1)
    # Concentric circles + diagonal thin lines for aliasing inspection.
    for radius in range(40, 340, 20):
        draw.ellipse([w // 2 - radius, h // 2 - radius, w // 2 + radius, h // 2 + radius],
                     outline=(255, 255, 255), width=1)
    draw.line([(0, h), (w, 0)], fill=(255, 255, 0), width=2)

    draw.text((48, 48), "PicPolish  sharpness / gradient chart", font=_font(44),
              fill=(255, 255, 255))
    draw.text((48, 112), "1280 x 720 - 1px grid - concentric rings - fine text",
              font=_font(24), fill=(230, 230, 230))
    draw.text((48, h - 96), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", font=_font(30), fill=(255, 255, 255))
    draw.text((48, h - 52), "abcdefghijklmnopqrstuvwxyz 0123456789 !@#$%^&*()",
              font=_font(26), fill=(240, 240, 240))

    path = SAMPLES_DIR / "text_gradient_1280x720.png"
    _save_png(im, path)
    manifest.append({"name": "text_gradient_1280x720.png", "kind": "synthetic",
                     "source": "generated", "width": w, "height": h,
                     "note": "渐变+文字+1px 网格；主观清晰度对比与锐度指标"})
    print(f"  [synth ] text_gradient_1280x720.png         1280x720")


def make_gradient(manifest: list[dict]) -> None:
    """Near-uniform gentle gradient: multi-tile seam / block-integrity probe.

    The colour range across the whole picture is deliberately tiny (a few
    levels), so every tile -- and every quadrant of the output -- must have
    almost the same mean.  Any broken/black tile or visible seam therefore
    stands out numerically (see the tile checks in ``e2e_windows.py``).
    """
    w = h = 700
    t = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    u = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.round(124.0 + 8.0 * t + 2.0 * u).astype(np.uint8)   # 124..134
    rgb[..., 1] = np.round(126.0 + 4.0 * u + 3.0 * t).astype(np.uint8)   # 126..133
    rgb[..., 2] = np.round(130.0 - 6.0 * t + 4.0 * u).astype(np.uint8)   # 124..134
    im = Image.fromarray(rgb, mode="RGB")
    path = SAMPLES_DIR / "solid_gradient_700x700.png"
    _save_png(im, path)
    manifest.append({"name": "solid_gradient_700x700.png", "kind": "synthetic",
                     "source": "generated", "width": w, "height": h,
                     "note": "近均匀渐变；700x700→1080x1080 多 tile 推理，测分块黑块/接缝"})
    print(f"  [synth ] solid_gradient_700x700.png        700x700")


def make_anime(manifest: list[dict]) -> None:
    """Anime-style synthetic frame: flat colour blocks + bold outlines + gradient.

    Used by the model matrix to compare anime-tuned models (anime6b / Real-CUGAN)
    against the general photo model (x4plus).  Deliberately has large flat areas
    and thick ink lines -- exactly where anime models should differ most.
    """
    w, h = 1280, 720
    t = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    u = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(40 + 110 * u + 40 * t, 0, 255)
    rgb[..., 1] = np.clip(95 + 120 * u + 25 * t, 0, 255)
    rgb[..., 2] = np.clip(205 - 45 * u + 25 * (1 - t), 0, 255)
    im = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(im)
    ink = (24, 22, 36)
    bold = 6

    # Sun + flat clouds.
    draw.ellipse([70, 60, 310, 300], fill=(255, 226, 120), outline=ink, width=bold)
    for cx, cy, rw, rh in ((540, 118, 150, 44), (790, 176, 186, 52), (1075, 96, 118, 38)):
        draw.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=(250, 250, 255),
                     outline=ink, width=5)

    # Flat-colour mountains with bold outlines (large blocks).
    draw.polygon([(0, 600), (210, 330), (420, 600)], fill=(94, 158, 124), outline=ink, width=bold)
    draw.polygon([(300, 600), (610, 260), (930, 600)], fill=(70, 132, 176), outline=ink, width=bold)
    draw.polygon([(820, 600), (1080, 350), (1280, 600)], fill=(126, 110, 190),
                 outline=ink, width=bold)

    # Anime character: big head, hair, huge eyes, flat shirt -- all bold inked.
    skin = (255, 224, 196)
    hair = (86, 58, 122)
    shirt = (238, 108, 122)
    draw.ellipse([736, 232, 964, 470], fill=skin, outline=ink, width=7)           # head
    draw.polygon([(716, 330), (752, 214), (860, 178), (968, 218), (1006, 340),
                  (962, 292), (908, 264), (842, 262), (784, 294), (746, 336)],
                 fill=hair, outline=ink, width=bold)                             # hair
    draw.polygon([(716, 330), (700, 470), (756, 460), (760, 356)],
                 fill=hair, outline=ink, width=bold)                             # side lock L
    draw.polygon([(1006, 340), (1024, 470), (968, 458), (958, 360)],
                 fill=hair, outline=ink, width=bold)                             # side lock R
    for cx in (816, 884):                                                        # eyes
        draw.ellipse([cx - 42, 322, cx + 42, 410], fill=(255, 255, 255),
                     outline=ink, width=6)
        draw.ellipse([cx - 20, 344, cx + 20, 392], fill=(58, 118, 214))
        draw.ellipse([cx - 8, 350, cx + 2, 364], fill=(255, 255, 255))
    draw.arc([826, 404, 874, 438], start=200, end=340, fill=ink, width=5)        # mouth
    draw.ellipse([748, 400, 796, 430], fill=(255, 168, 168), outline=ink, width=3)
    draw.ellipse([904, 400, 952, 430], fill=(255, 168, 168), outline=ink, width=3)
    draw.polygon([(772, 462), (928, 462), (1006, 720), (694, 720)],
                 fill=shirt, outline=ink, width=7)                               # body/shirt
    draw.polygon([(792, 466), (906, 466), (890, 512), (808, 512)],
                 fill=(244, 244, 252), outline=ink, width=4)                     # collar

    # Foreground grass band + anime "speed lines" in the sky.
    draw.polygon([(0, 628), (1280, 604), (1280, 720), (0, 720)],
                 fill=(88, 178, 96), outline=ink, width=6)
    for x in range(40, 1280, 150):
        draw.line([(x, 666), (x + 70, 646)], fill=(46, 120, 60), width=4)
    for x0, y0 in ((1180, 40), (1230, 90), (1130, 96), (40, 40), (0, 120)):
        draw.line([(x0, y0), (x0 - 90, y0 + 70)], fill=(255, 250, 210), width=4)

    # Speech bubble with text (crisp edges + flat fill).
    draw.rounded_rectangle([470, 60, 800, 176], radius=26, fill=(255, 255, 255),
                           outline=ink, width=6)
    draw.polygon([(520, 176), (560, 176), (520, 226)], fill=(255, 255, 255),
                 outline=ink, width=4)
    draw.text((500, 88), "SUPER RES!", font=_font(52), fill=(34, 30, 48))
    draw.text((500, 146), "anime / illustration test frame", font=_font(20),
              fill=(90, 90, 110))

    path = SAMPLES_DIR / "anime_1280x720.png"
    _save_png(im, path)
    manifest.append({"name": "anime_1280x720.png", "kind": "synthetic",
                     "source": "generated", "width": w, "height": h,
                     "note": "动漫风格合成帧（大色块+粗线条+渐变）；anime6b / CUGAN 与 x4plus 差异对比"})
    print("  [synth ] anime_1280x720.png                 1280x720")


def make_noisy(photos: dict[str, Image.Image], manifest: list[dict]) -> None:
    """Add Gaussian / salt-and-pepper noise to a real photo (denoise probes)."""
    cliff = photos["photo_cliff_1920x1200.jpg"]
    clean = _fit(cliff, (1280, 720))
    w, h = clean.size
    rng = np.random.default_rng(20261019)
    base = np.asarray(clean, dtype=np.float32)

    gauss = np.clip(base + rng.normal(0.0, 18.0, base.shape), 0, 255).astype(np.uint8)
    _save_jpeg(Image.fromarray(gauss, mode="RGB"), SAMPLES_DIR / "noisy_gauss_1280x720.jpg")
    manifest.append({"name": "noisy_gauss_1280x720.jpg", "kind": "derived",
                     "source": "photo_cliff_1920x1200.jpg", "width": w, "height": h,
                     "note": "真实照片 + 高斯噪声 σ=18；验证 rrdb denoise=false/true 与 realcugan denoise_level 差异"})
    print("  [noisy ] noisy_gauss_1280x720.jpg           1280x720  (gaussian sigma=18)")

    saltpepper = np.asarray(clean).copy()
    mask = rng.random((h, w))
    saltpepper[mask < 0.01] = 0
    saltpepper[mask > 0.99] = 255
    _save_jpeg(Image.fromarray(saltpepper, mode="RGB"),
               SAMPLES_DIR / "noisy_saltpepper_1280x720.jpg")
    manifest.append({"name": "noisy_saltpepper_1280x720.jpg", "kind": "derived",
                     "source": "photo_cliff_1920x1200.jpg", "width": w, "height": h,
                     "note": "真实照片 + 椒盐噪声 2%；降噪鲁棒性样本"})
    print("  [noisy ] noisy_saltpepper_1280x720.jpg      1280x720  (salt&pepper 2%)")


def _lap_var(im: Image.Image) -> float:
    gray = np.asarray(im.convert("L"), dtype=np.float64)
    lap = (4.0 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1]
           - gray[1:-1, :-2] - gray[1:-1, 2:])
    return float(lap.var())


def verify(manifest: list[dict]) -> int:
    total = 0
    print("\n验证生成结果：")
    for entry in manifest:
        path = SAMPLES_DIR / entry["name"]
        if not path.is_file():
            print(f"  FAIL 缺失 {entry['name']}")
            return 1
        with Image.open(path) as im:
            im.load()
            if (im.width, im.height) != (entry["width"], entry["height"]):
                print(f"  FAIL {entry['name']} 尺寸 {im.width}x{im.height} != "
                      f"{entry['width']}x{entry['height']}")
                return 1
            if entry["name"] == "alpha_800x600.png" and "A" not in im.getbands():
                print(f"  FAIL {entry['name']} 缺少 alpha 通道（mode={im.mode}）")
                return 1
        total += path.stat().st_size
    total_mb = total / (1024 * 1024)
    print(f"  全部 {len(manifest)} 个文件 OK，总计 {total_mb:.2f} MB（预算 {TOTAL_BUDGET_MB} MB）")
    if total_mb >= TOTAL_BUDGET_MB:
        print("  FAIL 超出总量预算")
        return 1

    # Extra sanity: the noisy probes must really be noisier than the clean photo.
    clean_path = SAMPLES_DIR / "photo_1280x720.jpg"
    if clean_path.is_file():
        with Image.open(clean_path) as clean_im:
            clean_lap = _lap_var(clean_im)
        for name in ("noisy_gauss_1280x720.jpg", "noisy_saltpepper_1280x720.jpg"):
            path = SAMPLES_DIR / name
            if not path.is_file():
                continue
            with Image.open(path) as noisy_im:
                noisy_lap = _lap_var(noisy_im)
            if noisy_lap <= clean_lap * 1.05:
                print(f"  FAIL {name} 噪声不够明显（lapvar {noisy_lap:.0f} vs clean {clean_lap:.0f}）")
                return 1
            print(f"  noise sanity: {name} lapvar {noisy_lap:.0f} > clean {clean_lap:.0f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PicPolish E2E samples")
    parser.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    args = parser.parse_args()

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    print(f"目标目录：{SAMPLES_DIR}")
    print("复制真实照片：")
    photos = make_real_photos(manifest)
    print("生成派生图：")
    make_derived(photos, manifest)
    print("生成合成图：")
    make_alpha(manifest)
    make_text_gradient(manifest)
    make_gradient(manifest)
    make_anime(manifest)
    print("生成降噪噪声样本：")
    make_noisy(photos, manifest)
    code = verify(manifest)
    manifest_path = SAMPLES_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"清单已写入 {manifest_path.name}")
    return code


if __name__ == "__main__":
    sys.exit(main())
