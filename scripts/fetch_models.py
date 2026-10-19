#!/usr/bin/env python3
"""Download / verify every model weight used by PicPolish (idempotent).

Assets (see ``app/models.py`` for the registry that consumes them):

======================  ========================================================
``RealESRGAN_x4plus.pth``            Real-ESRGAN x4plus (auto/realesrgan-x4plus)
``RealESRGAN_x2plus.pth``            Real-ESRGAN x2plus (auto/realesrgan-x2plus)
``RealESRGAN_x4plus_anime_6B.pth``   Real-ESRGAN anime (realesrgan-anime6b)
``realcugan/*.pth`` (11 files)       Real-CUGAN v3 (realcugan / -conservative),
                                     unpacked from the official
                                     ``updated_weights.zip`` release asset
======================  ========================================================

Design notes
------------
* **Idempotent**: a file whose size matches the published byte size is left
  untouched, so existing weights are never re-downloaded.
* **Atomic**: downloads go to ``<name>.part`` and are renamed only after the
  exact size was verified; a failed/interrupted transfer can never corrupt an
  existing good file.  The CUGAN zip is unpacked file-by-file through temp
  files and then atomically moved into ``models/realcugan/``.
* **Mirrors**: GitHub release assets are primary; ``ghfast.top`` (GitHub proxy)
  is tried next and ``hf-mirror.com`` (Hugging Face mirror, x4plus only) last.
* Only stdlib is used -- no torch, no network at import time.

Usage
-----
    python scripts/fetch_models.py                      # download missing weights
    python scripts/fetch_models.py --check              # verify only, exit 1 if missing
    python scripts/fetch_models.py --only realcugan     # fetch one asset group
    python scripts/fetch_models.py --force              # re-download even if present
    python scripts/fetch_models.py --models-dir DIR     # custom target directory

Exit codes: 0 = all selected weights present/valid, 1 = something missing/failed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # keep the script usable even if app/ is broken/missing
    from app import config

    DEFAULT_MODELS_DIR: Path = Path(config.MODELS_DIR)
except Exception:  # pragma: no cover - defensive
    DEFAULT_MODELS_DIR = ROOT / "models"

_CHUNK = 1024 * 1024
_USER_AGENT = "PicPolish/1.1 (scripts/fetch_models.py)"
#: Fallback sanity threshold when a published byte size is unknown.
MIN_FILE_BYTES = 1_000_000

_ESRGAN_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
_CUGAN_RELEASES = "https://github.com/bilibili/ailab/releases/download/Real-CUGAN"
_GHFAST = "https://ghfast.top/"


def _mirrors(github_url: str) -> list[str]:
    """GitHub primary, then the ``ghfast.top`` proxy prefix."""
    return [github_url, _GHFAST + github_url]


#: Real-CUGAN v3 checkpoints shipped inside ``updated_weights.zip``.
#: filename -> exact byte size (verified against the official release).
_CUGAN_FILES: dict[str, int] = {
    "up2x-latest-no-denoise.pth": 5147249,
    "up2x-latest-denoise1x.pth": 5147249,
    "up2x-latest-denoise2x.pth": 5147249,
    "up2x-latest-denoise3x.pth": 5147249,
    "up2x-latest-conservative.pth": 5147249,
    "up3x-latest-no-denoise.pth": 5154161,
    "up3x-latest-denoise3x.pth": 5154161,
    "up3x-latest-conservative.pth": 5154161,
    "up4x-latest-no-denoise.pth": 5636403,
    "up4x-latest-denoise3x.pth": 5636403,
    "up4x-latest-conservative.pth": 5636403,
}

_CUGAN_ZIP_URL = f"{_CUGAN_RELEASES}/updated_weights.zip"

ASSETS: tuple[dict, ...] = (
    {
        "id": "x4plus",
        "kind": "file",
        "label": "Real-ESRGAN x4plus（通用照片 4x）",
        "filename": "RealESRGAN_x4plus.pth",
        "size": 67040989,
        "urls": _mirrors(
            f"{_ESRGAN_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth"
        ) + [
            # Hugging Face mirror (verified same byte size).
            "https://hf-mirror.com/amd/realesrgan-x4plus/resolve/main/"
            "RealESRGAN_x4plus.pth",
        ],
    },
    {
        "id": "x2plus",
        "kind": "file",
        "label": "Real-ESRGAN x2plus（低倍率 2x）",
        "filename": "RealESRGAN_x2plus.pth",
        "size": 67061725,
        "urls": _mirrors(f"{_ESRGAN_RELEASES}/v0.2.1/RealESRGAN_x2plus.pth"),
    },
    {
        "id": "anime6b",
        "kind": "file",
        "label": "Real-ESRGAN anime6B（动漫 4x）",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "size": 17938799,
        "urls": _mirrors(
            f"{_ESRGAN_RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        ),
    },
    {
        "id": "realcugan",
        "kind": "zip",
        "label": "Real-CUGAN v3（11 个多档降噪权重）",
        "zip_filename": "updated_weights.zip",
        "size": 54001450,
        "urls": _mirrors(_CUGAN_ZIP_URL),
        # relative path under models/ -> expected byte size
        "targets": {
            f"realcugan/{name}": size for name, size in _CUGAN_FILES.items()
        },
    },
)

_ASSET_IDS = tuple(asset["id"] for asset in ASSETS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_state(path: Path, expected_size: int) -> tuple[str, int]:
    """Return ``("ok"|"missing"|"partial", size)`` for one weight file."""
    try:
        if not path.is_file():
            return "missing", 0
        size = path.stat().st_size
    except OSError:  # pragma: no cover - permissions etc.
        return "missing", 0
    if expected_size > 0:
        return ("ok" if size == expected_size else "partial"), size
    return ("ok" if size >= MIN_FILE_BYTES else "partial"), size


def _targets_of(asset: dict) -> dict[Path, int]:
    """Map of relative path -> expected size for every file of an asset."""
    if asset["kind"] == "file":
        return {Path(asset["filename"]): int(asset["size"])}
    return {Path(rel): int(size) for rel, size in asset["targets"].items()}


def _asset_state(asset: dict, models_dir: Path) -> tuple[bool, list[tuple[Path, str, int, int]]]:
    """Return ``(all_ok, [(path, state, size, expected), ...])``."""
    rows = []
    all_ok = True
    for rel, expected in _targets_of(asset).items():
        state, size = _file_state(models_dir / rel, expected)
        if state != "ok":
            all_ok = False
        rows.append((rel, state, size, expected))
    return all_ok, rows


def _download(url: str, dest: Path, *, timeout: float, expected_size: int = 0) -> None:
    """Download ``url`` to ``dest`` atomically; raises on any failure."""
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    started = time.monotonic()
    received = 0
    total = expected_size
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            try:
                content_length = int(response.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                content_length = 0
            if content_length:
                total = content_length
            with open(part, "wb") as fh:
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    received += len(chunk)
                    elapsed = max(time.monotonic() - started, 1e-6)
                    rate = received / elapsed
                    if total > 0:
                        percent = min(100.0, received * 100.0 / total)
                        text = (
                            f"\r  {percent:5.1f}%  {_human(received)} / {_human(total)}"
                            f"  {_human(rate)}/s"
                        )
                    else:
                        text = f"\r  {_human(received)}  {_human(rate)}/s"
                    print(text, end="", flush=True)
        print()
        if expected_size > 0 and received != expected_size:
            raise RuntimeError(
                f"字节数不符：收到 {received}，期望 {expected_size}（疑似下载不完整或被代理篡改）"
            )
        if received < MIN_FILE_BYTES:
            raise RuntimeError(
                f"下载内容过小（{received} 字节 < {_human(MIN_FILE_BYTES)}），疑似失败"
            )
        os.replace(part, dest)
    except BaseException:
        print()
        try:
            part.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - defensive
            pass
        raise


def _download_with_mirrors(
    urls: list[str], dest: Path, *, timeout: float, expected_size: int
) -> bool:
    """Try every mirror (twice each); return True when one download succeeded."""
    for index, url in enumerate(urls, start=1):
        for attempt in (1, 2):
            try:
                print(f"  [镜像 {index}/{len(urls)}] {url}")
                _download(url, dest, timeout=timeout, expected_size=expected_size)
                return True
            except KeyboardInterrupt:
                print("\n已取消下载；现有文件未被修改。")
                raise
            except Exception as exc:  # noqa: BLE001 - try next mirror/attempt
                print(f"  下载失败（第 {attempt}/2 次）：{exc}")
                if attempt == 1:
                    time.sleep(1.5)
    return False


def _fetch_file_asset(asset: dict, models_dir: Path, *, timeout: float, force: bool) -> bool:
    path = models_dir / asset["filename"]
    expected = int(asset["size"])
    state, size = _file_state(path, expected)
    if state == "ok" and not force:
        print(f"[{asset['id']}] 已存在（{_human(size)}），跳过")
        return True
    if state == "partial" and not force:
        print(
            f"[{asset['id']}] 现有文件不完整（{_human(size)} / 期望 {_human(expected)}），"
            "将原子替换"
        )
    elif force:
        print(f"[{asset['id']}] --force：重新下载覆盖")
    print(f"[{asset['id']}] 下载 {asset['label']}")
    if not _download_with_mirrors(
        list(asset["urls"]), path, timeout=timeout, expected_size=expected
    ):
        print(f"[{asset['id']}] [ERROR] 所有镜像均失败；可手动下载后放到：{path}")
        print(f"  官方地址：{asset['urls'][0]}")
        return False
    print(f"[{asset['id']}] 完成：{_human(path.stat().st_size)} → {path}")
    return True


def _extract_targets_from_zip(
    zip_path: Path, asset: dict, models_dir: Path, missing: dict[Path, int]
) -> bool:
    """Unpack only the expected ``.pth`` members, atomically per file."""
    ok = True
    extracted = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = {
            Path(info.filename).name: info for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".pth")
        }
        for rel, expected in missing.items():
            target = models_dir / rel
            info = members.get(target.name)
            if info is None:
                print(f"  [ERROR] zip 中缺少 {target.name}")
                ok = False
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            try:
                with archive.open(info) as source, open(tmp, "wb") as output:
                    shutil.copyfileobj(source, output, _CHUNK)
                size = tmp.stat().st_size
                if size != expected:
                    raise RuntimeError(
                        f"{target.name} 字节数不符（{size} != {expected}）"
                    )
                os.replace(tmp, target)
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:  # pragma: no cover - defensive
                    pass
            extracted += 1
    if extracted:
        print(f"[{asset['id']}] 已解压 {extracted} 个权重到 {models_dir / 'realcugan'}")
    return ok


def _fetch_zip_asset(asset: dict, models_dir: Path, *, timeout: float, force: bool) -> bool:
    all_ok, rows = _asset_state(asset, models_dir)
    if all_ok and not force:
        print(f"[{asset['id']}] {len(rows)}/{len(rows)} 个权重已存在，跳过")
        return True
    missing = {
        rel: expected for rel, state, _size, expected in rows
        if force or state != "ok"
    }
    print(
        f"[{asset['id']}] 下载 {asset['label']}"
        f"（需要 {len(missing)}/{len(rows)} 个文件）"
    )
    zip_path = models_dir / f".{asset['id']}.zip"
    try:
        if not _download_with_mirrors(
            list(asset["urls"]), zip_path, timeout=timeout, expected_size=int(asset["size"])
        ):
            print(
                f"[{asset['id']}] [ERROR] 所有镜像均失败；可手动下载 "
                f"{asset['urls'][0]} 并解压到 {models_dir / 'realcugan'}"
            )
            return False
        try:
            return _extract_targets_from_zip(zip_path, asset, models_dir, missing)
        except zipfile.BadZipFile as exc:
            print(f"[{asset['id']}] [ERROR] 下载的压缩包损坏：{exc}")
            return False
    finally:
        try:
            zip_path.unlink(missing_ok=True)
            zip_path.with_name(zip_path.name + ".part").unlink(missing_ok=True)
        except OSError:  # pragma: no cover - defensive
            pass


def _ensure_asset(asset: dict, models_dir: Path, *, timeout: float, force: bool) -> bool:
    if asset["kind"] == "file":
        return _fetch_file_asset(asset, models_dir, timeout=timeout, force=force)
    return _fetch_zip_asset(asset, models_dir, timeout=timeout, force=force)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="下载/校验 PicPolish 所需的全部模型权重（幂等，支持多镜像）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查权重是否存在且完整，不下载；缺失时退出码为 1",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使文件已存在也重新下载（先写 .part 再原子替换）",
    )
    parser.add_argument(
        "--only",
        default="",
        help=f"只处理指定资产（逗号分隔，可选 {', '.join(_ASSET_IDS)}，默认全部）",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="权重目录（默认取 app.config.MODELS_DIR）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="单次网络读取超时（秒）",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="额外打印每个权重文件的 sha256",
    )
    return parser


def _select_assets(only: str) -> list[dict]:
    if not only or not only.strip() or only.strip().lower() == "all":
        return list(ASSETS)
    wanted = [part.strip().lower() for part in only.split(",") if part.strip()]
    unknown = [part for part in wanted if part not in _ASSET_IDS]
    if unknown:
        raise SystemExit(
            f"未知的资产：{', '.join(unknown)}（可选 {', '.join(_ASSET_IDS)}）"
        )
    return [asset for asset in ASSETS if asset["id"] in wanted]


def _safe_console() -> None:
    """Never crash on legacy Windows code pages (GBK) when printing.

    When stdout is a real console Python writes UTF-16 through the Windows
    console API, so only piped/redirected output is affected: use UTF-8 there
    (matching PowerShell 7 and the ``start.bat`` log redirection) and never
    raise on characters the active code page cannot encode.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - non-standard stream
            pass


def _print_check(asset: dict, models_dir: Path, *, show_hash: bool) -> bool:
    all_ok, rows = _asset_state(asset, models_dir)
    if all_ok:
        total = sum(expected for _rel, _s, _size, expected in rows)
        extra = ""
        if show_hash:
            hashes = [
                f"{rel.name}={_sha256(models_dir / rel)[:12]}…"
                for rel, _s, _size, _e in rows[:2]
            ]
            extra = " " + " ".join(hashes)
        print(f"  [OK]      {asset['id']:<12} {len(rows)}/{len(rows)} 个文件，"
              f"{_human(total)}{extra}")
        return True
    ok_count = sum(1 for _rel, state, _size, _e in rows if state == "ok")
    missing = [rel.name for rel, state, _size, _e in rows if state != "ok"]
    detail = ", ".join(missing[:3]) + ("…" if len(missing) > 3 else "")
    print(f"  [MISSING] {asset['id']:<12} {ok_count}/{len(rows)} 个文件就绪；缺 {detail}")
    if show_hash:
        for rel, state, _size, _e in rows:
            if state == "ok":
                print(f"              sha256({rel.name}) = {_sha256(models_dir / rel)}")
    return False


def main(argv: list[str] | None = None) -> int:
    _safe_console()
    args = _build_parser().parse_args(argv)
    models_dir: Path = args.models_dir.expanduser().resolve()
    selected = _select_assets(args.only)

    if args.check:
        print(f"检查模型权重目录：{models_dir}")
        all_ok = True
        for asset in selected:
            if not _print_check(asset, models_dir, show_hash=args.hash):
                all_ok = False
        if not all_ok:
            print("\n缺少权重：请运行 `python scripts/fetch_models.py` 下载。")
            return 1
        print("全部权重就绪。")
        return 0

    models_dir.mkdir(parents=True, exist_ok=True)
    needed = 0
    for asset in selected:
        all_ok, rows = _asset_state(asset, models_dir)
        if not all_ok or args.force:
            needed += int(asset["size"])
    try:
        free = shutil.disk_usage(models_dir).free
    except OSError:  # pragma: no cover - weird filesystems
        free = None
    if free is not None and needed and free < needed + 64 * 1024 * 1024:
        print(f"[WARN] 磁盘剩余 {_human(free)}，预计需要下载 {_human(needed)}，可能失败。")

    print(f"模型权重目录：{models_dir}")
    ok = True
    for asset in selected:
        if not _ensure_asset(asset, models_dir, timeout=args.timeout, force=args.force):
            ok = False
    if not ok:
        return 1
    print("\n全部权重就绪，可启动 PicPolish。")
    if args.hash:
        for asset in selected:
            for rel, expected in _targets_of(asset).items():
                path = models_dir / rel
                if path.is_file() and path.stat().st_size == expected:
                    print(f"sha256({rel.as_posix()}) = {_sha256(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
