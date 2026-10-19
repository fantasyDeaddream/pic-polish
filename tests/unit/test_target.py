"""Unit tests for :mod:`app.target` (output-size planner).

The expectations below are derived from ``docs/API.md`` §1 ``目标分辨率档位``
(especially the rule list in §1.1 and its example table), **not** from reading
the implementation, so a regression in ``app/target.py`` is caught here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # allow `pytest tests/unit/...` from any cwd
    sys.path.insert(0, str(ROOT))

from app import config
from app.target import TIERS, InvalidTarget, get_tier, plan_many, plan_target

# ---------------------------------------------------------------------------
# API.md §1.1 example table: the authoritative rows
# ---------------------------------------------------------------------------
API_TABLE = [
    # width, height, tier, expected_w, expected_h, expected_scale, expected_mode
    (1280, 720, "1080p", 1920, 1080, 1.5, "upscale"),
    (1920, 1080, "2k", 2560, 1440, 2560 / 1920, "upscale"),
    (1000, 667, "1080p", 1616, 1080, 1080 / 667, "upscale"),
    (1080, 1920, "720p", 720, 1280, 720 / 1080, "downscale"),
    (3840, 2160, "1080p", 1920, 1080, 1080 / 2160, "downscale"),
]

PLAN_KEYS = {
    "tier",
    "label",
    "rule",
    "value",
    "origin",
    "output",
    "scale",
    "mode",
    "aligned",
    "note",
}


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------
def test_tiers_definition():
    assert list(TIERS) == ["720p", "1080p", "2k", "4k"]
    expected = {
        "720p": ("short", 720),
        "1080p": ("short", 1080),
        "2k": ("long", 2560),
        "4k": ("long", 3840),
    }
    for tier_id, (rule, value) in expected.items():
        tier = TIERS[tier_id]
        assert tier["id"] == tier_id
        assert tier["rule"] == rule
        assert tier["value"] == value
        assert tier["label"] and isinstance(tier["label"], str)
        assert tier["hint"] and isinstance(tier["hint"], str)


def test_get_tier_returns_a_copy():
    tier = get_tier("1080p")
    assert tier == dict(TIERS["1080p"])
    tier["value"] = 1  # mutating the copy must not poison the global table
    assert TIERS["1080p"]["value"] == 1080


@pytest.mark.parametrize("bad", [None, "", "1080P", "4K", "720", "foo", 1080, 2.0])
def test_get_tier_rejects_unknown_tiers(bad):
    with pytest.raises(InvalidTarget):
        get_tier(bad)
    # InvalidTarget is a ValueError, so FastAPI-style handling stays simple.
    with pytest.raises(ValueError):
        get_tier(bad)


# ---------------------------------------------------------------------------
# API.md §1.1 example table
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "width,height,tier,out_w,out_h,scale,mode", API_TABLE
)
def test_api_table_rows(width, height, tier, out_w, out_h, scale, mode):
    plan = plan_target(width, height, tier)
    assert plan["origin"] == {"width": width, "height": height}
    assert plan["output"] == {"width": out_w, "height": out_h}
    assert plan["mode"] == mode
    assert plan["scale"] == pytest.approx(round(scale, 6), abs=1e-6)
    assert plan["tier"] == tier
    assert plan["label"] == TIERS[tier]["label"]
    assert plan["rule"] == TIERS[tier]["rule"]
    assert plan["value"] == TIERS[tier]["value"]
    assert isinstance(plan["aligned"], bool)
    assert isinstance(plan["note"], str) and plan["note"]


def test_plan_has_exactly_the_contract_keys():
    plan = plan_target(1280, 720, "1080p")
    assert set(plan) == PLAN_KEYS


def test_1000x667_alignment_is_flagged():
    # round(1000 * 1080/667) = 1619 -> snapped to 1616 (nearest multiple of 8)
    plan = plan_target(1000, 667, "1080p")
    assert plan["aligned"] is True
    assert plan["output"] == {"width": 1616, "height": 1080}
    assert abs(plan["output"]["width"] - 1619) <= config.DEFAULT_ALIGN // 2
    assert "对齐" in plan["note"]


# ---------------------------------------------------------------------------
# Additional contract rows (derived from §1.1 rules)
# ---------------------------------------------------------------------------
EXTRA_ROWS = [
    # landscape / portrait / square, short and long rules
    (1280, 720, "720p", 1280, 720, 1.0, "same"),
    (1920, 1080, "4k", 3840, 2160, 2.0, "upscale"),
    (1080, 1920, "2k", 1440, 2560, 2560 / 1920, "upscale"),
    (720, 1280, "1080p", 1080, 1920, 1080 / 720, "upscale"),
    (1000, 1000, "1080p", 1080, 1080, 1080 / 1000, "upscale"),
    (1000, 1000, "4k", 3840, 3840, 3840 / 1000, "upscale"),
    (320, 240, "4k", 3840, 2880, 3840 / 320, "upscale"),
    (3840, 2160, "4k", 3840, 2160, 1.0, "same"),
    (1000, 667, "2k", 2560, 1712, 2560 / 1000, "upscale"),
    (1280, 720, "1080p", 1920, 1080, 1.5, "upscale"),
]


@pytest.mark.parametrize("width,height,tier,out_w,out_h,scale,mode", EXTRA_ROWS)
def test_extra_rows(width, height, tier, out_w, out_h, scale, mode):
    plan = plan_target(width, height, tier)
    assert plan["output"] == {"width": out_w, "height": out_h}
    assert plan["mode"] == mode
    assert plan["scale"] == pytest.approx(round(scale, 6), abs=1e-6)


@pytest.mark.parametrize("width,height,tier", [(1280, 720, "1080p"), (1920, 1080, "2k"), (1080, 1920, "720p"), (1000, 667, "1080p"), (1000, 1000, "4k")])
def test_benchmark_side_is_exact_and_free_side_is_aligned(width, height, tier):
    """The benchmark side is fixed exactly; the other side snaps to `align`."""
    plan = plan_target(width, height, tier, align=config.DEFAULT_ALIGN)
    value = plan["value"]
    if plan["rule"] == "short":
        assert min(plan["output"]["width"], plan["output"]["height"]) == value
    else:
        assert max(plan["output"]["width"], plan["output"]["height"]) == value
    out_w, out_h = plan["output"]["width"], plan["output"]["height"]
    assert out_w % config.DEFAULT_ALIGN == 0
    assert out_h % config.DEFAULT_ALIGN == 0
    # The free side deviates from the ideal ratio by at most align/2 pixels.
    if plan["rule"] == "short":
        free = out_w if width > height else out_h
        raw = round((width if width > height else height) * plan["scale"])
    else:
        free = out_h if width >= height else out_w
        raw = round((height if width >= height else width) * plan["scale"])
    assert abs(free - raw) <= config.DEFAULT_ALIGN // 2


@pytest.mark.parametrize(
    "align,expected_w",
    [(2, 1620), (4, 1620), (8, 1616), (16, 1616), (64, 1600)],
)
def test_align_parameter_snapping(align, expected_w):
    plan = plan_target(1000, 667, "1080p", align=align)
    assert plan["output"]["width"] == expected_w
    assert plan["output"]["height"] == 1080  # benchmark side is not snapped
    assert plan["aligned"] is True
    assert plan["output"]["width"] % align == 0


def test_align_2_keeps_benchmark_side_untouched():
    plan = plan_target(1280, 720, "1080p", align=2)
    assert plan["output"] == {"width": 1920, "height": 1080}
    assert plan["aligned"] is False


def test_align_boundaries_are_accepted():
    for align in (config.MIN_ALIGN, config.MAX_ALIGN):
        plan = plan_target(1280, 720, "1080p", align=align)
        assert plan["output"] == {"width": 1920, "height": 1080}


def test_no_alignment_flags_false():
    plan = plan_target(1280, 720, "1080p")
    assert plan["aligned"] is False
    assert "对齐" not in plan["note"]


def test_minimum_free_side_is_clamped_to_align():
    # A 16384x16 panorama at 2k: the short side would become ~2.5 px -> min 8 px.
    plan = plan_target(16384, 16, "2k")
    assert plan["output"] == {"width": 2560, "height": config.DEFAULT_ALIGN}


# ---------------------------------------------------------------------------
# Notes are user-facing Chinese strings
# ---------------------------------------------------------------------------
def test_notes_describe_the_rule():
    short_note = plan_target(1280, 720, "1080p")["note"]
    assert "短边" in short_note
    long_note = plan_target(1920, 1080, "2k")["note"]
    assert "长边" in long_note
    downscale_note = plan_target(3840, 2160, "1080p")["note"]
    assert "高质量缩放" in downscale_note
    same_note = plan_target(1280, 720, "720p")["note"]
    assert "整洁化" in same_note


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------
def test_unknown_tier_rejected():
    with pytest.raises(InvalidTarget):
        plan_target(1280, 720, "8k")


@pytest.mark.parametrize(
    "width,height",
    [
        (0, 720),
        (1280, 0),
        (-5, 720),
        (1280, -1),
        (8, 8),  # smaller than MIN_SIDE
        (config.MIN_SIDE - 1, 1000),
        (20000, 100),  # larger than MAX_SIDE
        (100, 20000),
        (5000, 5000),  # more pixels than MAX_PIXELS
        ("abc", 720),
        (1280, None),
    ],
)
def test_invalid_dimensions_rejected(width, height):
    with pytest.raises(InvalidTarget):
        plan_target(width, height, "1080p")


@pytest.mark.parametrize("align", [0, 1, config.MAX_ALIGN + 1, -8, 8.0])
def test_invalid_align_rejected(align):
    with pytest.raises(InvalidTarget):
        plan_target(1280, 720, "1080p", align=align)


def test_output_size_limit_rejected():
    # short side 16 px -> 720p would need 737280 px height, above MAX_SIDE
    with pytest.raises(InvalidTarget):
        plan_target(16, 16384, "720p")
    with pytest.raises(InvalidTarget):
        plan_target(100, 16384, "720p")


# ---------------------------------------------------------------------------
# plan_many
# ---------------------------------------------------------------------------
def test_plan_many_adds_index_and_name():
    images = [
        {"name": "a.jpg", "width": 1280, "height": 720},
        {"name": "b.png", "width": 1080, "height": 1920},
    ]
    plans = plan_many(images, "1080p")
    assert [p["index"] for p in plans] == [0, 1]
    assert [p["name"] for p in plans] == ["a.jpg", "b.png"]
    assert plans[0]["output"] == {"width": 1920, "height": 1080}
    assert plans[1]["output"] == {"width": 1080, "height": 1920}
    assert all(set(p) == PLAN_KEYS | {"index", "name"} for p in plans)


def test_plan_many_defaults_name():
    plans = plan_many([{"width": 640, "height": 480}], "720p")
    assert plans[0]["name"] == "image_1"
    assert plans[0]["index"] == 0


def test_plan_many_empty_list():
    assert plan_many([], "1080p") == []


def test_plan_many_invalid_entries():
    with pytest.raises(InvalidTarget):
        plan_many([{"name": "a.jpg", "height": 720}], "1080p")  # missing width
    with pytest.raises(InvalidTarget):
        plan_many(["not-a-dict"], "1080p")
    with pytest.raises(InvalidTarget):
        plan_many([{"name": "a.jpg", "width": 1280, "height": 720}], "bogus")


def test_plan_many_propagates_row_values():
    plans = plan_many(
        [{"name": f"img{i}.jpg", "width": 1000, "height": 667} for i in range(3)],
        "1080p",
    )
    assert len(plans) == 3
    for index, plan in enumerate(plans):
        assert plan["index"] == index
        assert plan["output"] == {"width": 1616, "height": 1080}
        assert plan["aligned"] is True
