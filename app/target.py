"""Target-resolution planning for PicPolish.

Tier rules (see ``docs/API.md`` section 1.1 for the authoritative description):

* ``720p`` / ``1080p`` -- the **short side** is fixed to 720 / 1080 px.
* ``2k`` / ``4k``       -- the **long side** is fixed to 2560 / 3840 px.

The original aspect ratio is preserved; the free side is rounded and then snapped
to the nearest multiple of ``align`` (default 8 px) so every output dimension is
codec/GPU friendly.  The deviation introduced by snapping is at most ``align/2``
pixels, which is the "细小差异" allowed by the product spec.

This module is provided by the Lead and is treated as read-only by the rest of
the code base.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from . import config


class InvalidTarget(ValueError):
    """Raised for an unknown tier or an invalid image dimension."""


TIERS: "OrderedDict[str, dict]" = OrderedDict(
    (
        ("720p", {"id": "720p", "label": "720P", "rule": "short", "value": 720,
                  "hint": "短边固定 720 像素"}),
        ("1080p", {"id": "1080p", "label": "1080P", "rule": "short", "value": 1080,
                   "hint": "短边固定 1080 像素"}),
        ("2k", {"id": "2k", "label": "2K", "rule": "long", "value": 2560,
                "hint": "长边固定 2560 像素"}),
        ("4k", {"id": "4k", "label": "4K", "rule": "long", "value": 3840,
                "hint": "长边固定 3840 像素"}),
    )
)


def get_tier(tier: str) -> dict:
    """Return a copy of the tier definition, raising :class:`InvalidTarget`."""
    if not isinstance(tier, str) or tier not in TIERS:
        raise InvalidTarget(f"未知的 target：{tier}（可选 720p/1080p/2k/4k）")
    return dict(TIERS[tier])


def _snap(value: int, align: int) -> int:
    """Snap ``value`` to the nearest multiple of ``align`` (ties round up)."""
    return max(align, int(round(value / align)) * align)


def _validate_dimensions(width: int, height: int) -> tuple[int, int]:
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise InvalidTarget("图片宽高必须是整数") from exc
    if width <= 0 or height <= 0:
        raise InvalidTarget(f"图片尺寸非法：{width}x{height}")
    if width < config.MIN_SIDE or height < config.MIN_SIDE:
        raise InvalidTarget(f"图片过小：{width}x{height}（最小边 {config.MIN_SIDE}px）")
    if width > config.MAX_SIDE or height > config.MAX_SIDE:
        raise InvalidTarget(f"图片过大：{width}x{height}（最大边 {config.MAX_SIDE}px）")
    if width * height > config.MAX_PIXELS:
        raise InvalidTarget(f"图片像素过多：{width * height}（上限 {config.MAX_PIXELS}）")
    return width, height


def plan_target(width: int, height: int, tier: str, align: int = 8) -> dict:
    """Compute the output size for one image.

    Returns a dict with keys: ``tier, label, rule, value, origin, output, scale,
    mode, aligned, note``.  Raises :class:`InvalidTarget` on bad input.
    """
    width, height = _validate_dimensions(width, height)
    if not isinstance(align, int) or not (config.MIN_ALIGN <= align <= config.MAX_ALIGN):
        raise InvalidTarget(f"align 必须是 {config.MIN_ALIGN}..{config.MAX_ALIGN} 的整数：{align!r}")
    tier_def = get_tier(tier)

    if tier_def["rule"] == "short":
        base = min(width, height)
    else:
        base = max(width, height)
    scale = tier_def["value"] / float(base)

    # Fix the benchmark side exactly, scale + snap the other side.
    aligned = False
    if tier_def["rule"] == "short":
        if width <= height:  # portrait / square: short side is width
            out_w = tier_def["value"]
            raw_other = int(round(height * scale))
            out_h = _snap(raw_other, align)
            aligned = out_h != raw_other
        else:  # landscape: short side is height
            out_h = tier_def["value"]
            raw_other = int(round(width * scale))
            out_w = _snap(raw_other, align)
            aligned = out_w != raw_other
    else:  # long side fixed
        if width >= height:  # landscape: long side is width
            out_w = tier_def["value"]
            raw_other = int(round(height * scale))
            out_h = _snap(raw_other, align)
            aligned = out_h != raw_other
        else:  # portrait: long side is height
            out_h = tier_def["value"]
            raw_other = int(round(width * scale))
            out_w = _snap(raw_other, align)
            aligned = out_w != raw_other

    if out_w > config.MAX_SIDE or out_h > config.MAX_SIDE:
        raise InvalidTarget(
            f"计算出的输出尺寸过大：{out_w}x{out_h}（最大边 {config.MAX_SIDE}px），请换更低档位"
        )
    if out_w * out_h > config.MAX_PIXELS:
        raise InvalidTarget(f"计算出的输出像素过多：{out_w * out_h}（上限 {config.MAX_PIXELS}）")

    if scale > 1.0 + 1e-9:
        mode = "upscale"
    elif scale < 1.0 - 1e-9:
        mode = "downscale"
    else:
        mode = "same"

    if tier_def["rule"] == "short":
        note = f"短边 {base} → {tier_def['value']}"
    else:
        note = f"长边 {base} → {tier_def['value']}"
    if width <= height and tier_def["rule"] == "short":
        note += f"；长边 {height} → {out_h}"
    elif width > height and tier_def["rule"] == "short":
        note += f"；长边 {width} → {out_w}"
    elif width >= height:
        note += f"；短边 {height} → {out_h}"
    else:
        note += f"；短边 {width} → {out_w}"
    if aligned:
        note += f"（已对齐 {align}px 网格）"
    if mode == "downscale":
        note += "；目标小于原图，采用高质量缩放"
    elif mode == "same":
        note += "；目标与原图相同，仅做整洁化处理"

    return {
        "tier": tier_def["id"],
        "label": tier_def["label"],
        "rule": tier_def["rule"],
        "value": tier_def["value"],
        "origin": {"width": width, "height": height},
        "output": {"width": out_w, "height": out_h},
        "scale": round(scale, 6),
        "mode": mode,
        "aligned": aligned,
        "note": note,
    }


def plan_many(images: Iterable[dict], tier: str, align: int = 8) -> list[dict]:
    """Plan several images at once.

    ``images`` items need ``name`` (optional) plus ``width``/``height``.
    Each returned dict additionally carries ``index`` and ``name``.
    """
    results: list[dict] = []
    for index, item in enumerate(images):
        if not isinstance(item, dict):
            raise InvalidTarget(f"第 {index + 1} 个图片信息格式错误")
        try:
            width = item["width"]
            height = item["height"]
        except KeyError as exc:
            raise InvalidTarget(f"第 {index + 1} 个图片缺少 {exc.args[0]} 字段") from exc
        plan = plan_target(width, height, tier, align=align)
        plan["index"] = index
        plan["name"] = str(item.get("name") or f"image_{index + 1}")
        results.append(plan)
    return results
