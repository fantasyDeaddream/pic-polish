#!/usr/bin/env bash
# =============================================================================
#  PicPolish Linux / WSL 运行环境准备脚本（幂等，可重复运行）
#
#  用法:
#      bash scripts/setup_env.sh
#
#  行为（首次启动 / 分发场景会自动被 start.sh 调用）:
#    1. .venv-linux 不存在 → 创建 venv（python3 -m venv；若系统缺 ensurepip 则用
#       pip --user --break-system-packages 安装 virtualenv 兜底，脚本会打印说明）；
#    2. 已存在 → 校验；缺 Web 依赖时补装 requirements-linux.txt（含 opencv），不重建 venv；
#    3. 没有 torch → 按本机实测有效的**离线配方**安装 CUDA 12.1 版 torch：
#         a) 找到本地 wheelhouse（$PIC_POLISH_WHEELHOUSE → ~/picpolish_wheels →
#            <项目>/wheelhouse → pip 缓存 ~/.cache/pip 中的 .whl 汇总目录）；
#         b) 用 Python zipfile 把 wheel 直接解包到 .venv-linux/lib/python3.12/site-packages
#            （绕过 pip 重新下载 2.9GB nvidia 依赖，本机实测数秒完成）；
#         c) pip 补装 wheelhouse 里缺少的小包（nvidia-cuda-runtime-cu12 / nvidia-nvtx-cu12 /
#            mpmath / filelock / jinja2 / fsspec）；
#         d) 没有 wheelhouse 或离线失败 → 联网安装（$PIC_POLISH_TORCH_INDEX，
#            默认 https://download.pytorch.org/whl/cu121，固定 torch 2.5.1 / torchvision 0.20.1），
#            最后尝试默认 PyPI；全部失败会提示“降级 pil”，不阻塞 Web 依赖安装；
#    4. 校验 torch / torch.cuda.is_available() / opencv，打印结果；
#    5. 检查模型权重状态（缺失时首次启动会自动调用 scripts/fetch_models.py 下载）。
#
#  WSL 提示：本机 CUDA 由 Windows 驱动直通（nvidia-smi 可见），安装 cu121 torch 后
#  `/api/health` 应显示 realesrgan/cuda。离线分发时只需额外提供 wheelhouse 目录。
#
#  环境变量:
#      PIC_POLISH_WHEELHOUSE     本地 wheelhouse 目录（离线安装 torch 优先使用）
#      PIC_POLISH_TORCH_INDEX    在线 torch 安装源（默认 cu121；CPU 机器可用 .../cpu）
#      PIC_POLISH_TORCH_VERSION  torch 版本（默认 2.5.1）
#      PIC_POLISH_TORCHVISION_VERSION  torchvision 版本（默认 0.20.1）
#      PIC_POLISH_SKIP_TORCH=1   跳过 torch 安装（强制使用 pil 兼容引擎）
#      PIC_POLISH_MODELS_DIR     模型权重目录（默认 <项目>/models）
#      PIC_POLISH_PYPI_INDEX     补装小依赖用的 PyPI 镜像（可选）
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-linux"
PY="$VENV/bin/python"
REQ="$ROOT/requirements-linux.txt"
FETCH_MODELS="$ROOT/scripts/fetch_models.py"
TORCH_INDEX="${PIC_POLISH_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
TORCH_VERSION="${PIC_POLISH_TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${PIC_POLISH_TORCHVISION_VERSION:-0.20.1}"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_TITLE=$'\033[36m'; C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_DIM=$'\033[2m'; C_RST=$'\033[0m'
else
    C_TITLE=''; C_OK=''; C_WARN=''; C_ERR=''; C_DIM=''; C_RST=''
fi
title() { printf '\n%s==== %s ====%s\n' "$C_TITLE" "$1" "$C_RST"; }
info()  { printf '  %s\n' "$1"; }
ok()    { printf '  %s[OK] %s%s\n' "$C_OK" "$1" "$C_RST"; }
warn()  { printf '  %s[警告] %s%s\n' "$C_WARN" "$1" "$C_RST"; }
fail()  { printf '  %s[错误] %s%s\n' "$C_ERR" "$1" "$C_RST"; }

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUTF8=1
export PIP_NO_INPUT=1

# ---------------------------------------------------------------------------
# venv 依赖探测
# ---------------------------------------------------------------------------
# 输出 JSON 报告；import 失败不算脚本错误
check_venv() {
    "$PY" - <<'PY'
import importlib, json, sys
report = {"python": sys.version.split()[0], "missing": [], "torch": None, "cuda": False, "gpu": None}
for name in ("fastapi", "uvicorn", "multipart", "PIL", "numpy", "cv2"):
    try:
        importlib.import_module(name)
    except Exception:
        report["missing"].append(name)
try:
    import torch
    report["torch"] = torch.__version__
    report["cuda"] = bool(torch.cuda.is_available())
    if report["cuda"]:
        try:
            report["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            report["gpu"] = None
except Exception as exc:
    report["torch_error"] = str(exc)
print(json.dumps(report, ensure_ascii=False))
PY
}

# 从 JSON 报告里取字段（字符串）
report_field() {
    local report="$1" field="$2"
    python3 - "$report" "$field" <<'PY'
import json, sys
try:
    data = json.loads(sys.argv[1])
except Exception:
    print("")
    raise SystemExit(0)
value = data.get(sys.argv[2])
print("" if value is None else value)
PY
}

report_missing_count() {
    local report="$1"
    python3 - "$report" <<'PY'
import json, sys
try:
    print(len(json.loads(sys.argv[1]).get("missing", [])))
except Exception:
    print("99")
PY
}

show_report() {
    local report="$1" python_ver torch_ver cuda gpu missing
    python_ver="$(report_field "$report" python)"
    torch_ver="$(report_field "$report" torch)"
    cuda="$(report_field "$report" cuda)"
    gpu="$(report_field "$report" gpu)"
    missing="$(python3 - "$report" <<'PY'
import json, sys
try:
    print(", ".join(json.loads(sys.argv[1]).get("missing", [])))
except Exception:
    print("?")
PY
)"
    info "Python:  $python_ver"
    if [ -n "$torch_ver" ]; then
        ok "torch:   $torch_ver（CUDA 可用: $cuda${gpu:+，GPU: $gpu}）"
    else
        warn "torch:   未安装（服务会降级到 pil 兼容引擎）"
    fi
    if [ -n "$missing" ]; then
        warn "缺少依赖: $missing"
    else
        ok "Web 依赖: fastapi / uvicorn / python-multipart / pillow / numpy / opencv(cv2) 均可用"
    fi
    # 报告里 cuda 是 Python bool 的字符串形式，单独再校验一次真实 CUDA
    if [ -n "$torch_ver" ]; then
        if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            ok 'CUDA:    torch.cuda.is_available() = True'
        else
            warn 'CUDA:    torch.cuda.is_available() = False（将以 CPU 运行 AI 超分，速度较慢；或降级 pil）'
        fi
    fi
}

# ---------------------------------------------------------------------------
# requirements 安装
# ---------------------------------------------------------------------------
install_requirements() {
    local wh="$1"
    info "安装 Web 依赖: $REQ"
    info "命令: $PY -m pip install -r requirements-linux.txt"
    if "$PY" -m pip install --disable-pip-version-check --no-input \
        --timeout 120 --retries 2 -r "$REQ"; then
        ok 'Web 依赖安装完成'
        return 0
    fi
    if [ -n "$wh" ]; then
        warn '常规安装失败，尝试使用本地 wheelhouse 补充...'
        if "$PY" -m pip install --disable-pip-version-check --no-input \
            --no-index --find-links "$wh" -r "$REQ"; then
            ok 'Web 依赖安装完成（本地 wheelhouse）'
            return 0
        fi
    fi
    fail 'Web 依赖安装失败（网络/镜像问题）。'
    info '可配置代理后重试，或把依赖 wheel 放入 wheelhouse 目录后重新运行：'
    info "  $PY -m pip install -r '$REQ'"
    return 1
}

# ---------------------------------------------------------------------------
# 本地 wheelhouse 探测（离线安装 torch 用）
# ---------------------------------------------------------------------------
find_wheelhouse() {
    local dir
    for dir in \
        "${PIC_POLISH_WHEELHOUSE:-}" \
        "$HOME/picpolish_wheels" \
        "$ROOT/wheelhouse" \
        "/home/deaddream/picpolish_wheels"; do
        [ -n "$dir" ] || continue
        [ -d "$dir" ] || continue
        if find "$dir" -maxdepth 1 -name '*.whl' -print -quit 2>/dev/null | grep -q .; then
            printf '%s\n' "$dir"
            return 0
        fi
    done
    # pip 下载缓存兜底：把 ~/.cache/pip 下的 .whl 汇总成临时 --find-links 目录
    if [ -d "$HOME/.cache/pip" ]; then
        local linkdir="${TMPDIR:-/tmp}/picpolish_pip_wheels"
        rm -rf "$linkdir"
        mkdir -p "$linkdir"
        local count=0 wheel
        while IFS= read -r wheel; do
            ln -sf "$wheel" "$linkdir/$(basename "$wheel")" 2>/dev/null && count=$((count + 1))
        done < <(find "$HOME/.cache/pip" -name '*.whl' -type f 2>/dev/null | head -300)
        if [ "$count" -gt 0 ]; then
            info "pip 缓存中找到 $count 个 wheel，汇总到 $linkdir（离线 --find-links）" >&2
            printf '%s\n' "$linkdir"
            return 0
        fi
    fi
    return 1
}

# ---------------------------------------------------------------------------
# torch 安装：优先本地 wheelhouse 免 pip 直装（离线解包），失败再联网，最后降级提示
# ---------------------------------------------------------------------------
torch_import_ok() {
    "$PY" -c "import torch" >/dev/null 2>&1
}

torch_cuda_ok() {
    "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1
}

venv_site_packages() {
    "$PY" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true
}

# 把 wheelhouse 里的 wheel 直接解包到 venv site-packages。
# 原因：pip 即使给了 --find-links，仍可能去镜像重新下载 2.9GB nvidia 依赖；
# 直接 zipfile 解包在本机实测只需数秒即可完成 torch 全家桶安装（Lead 已验证）。
extract_wheels_from_wheelhouse() {
    local wh="$1" site_packages
    site_packages="$(venv_site_packages)"
    if [ -z "$site_packages" ]; then
        fail '无法确定 venv site-packages 路径，无法离线解包'
        return 1
    fi
    info "离线解包 wheelhouse → $site_packages"
    "$PY" - "$wh" "$site_packages" <<'PY'
import glob, os, sys, zipfile

wheelhouse, site_packages = sys.argv[1], sys.argv[2]
os.makedirs(site_packages, exist_ok=True)
count = 0
skipped = []
failed = []
for wheel in sorted(glob.glob(os.path.join(wheelhouse, "*.whl"))):
    name = os.path.basename(wheel)
    # 只解包可用于当前 Python 3.12 的 wheel（纯 Python/abi3/cp312）
    if not any(tag in name for tag in ("cp312", "py3-none-any", "none-any", "abi3")):
        skipped.append(name)
        continue
    try:
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(site_packages)
        count += 1
    except Exception as exc:  # noqa: BLE001 - 记录后继续解包其他 wheel
        failed.append((name, str(exc)))
print("  已解包 %d 个 wheel" % count)
if skipped:
    print("  跳过 %d 个与当前 Python 不兼容的 wheel" % len(skipped))
for name, error in failed:
    print("  [警告] 解包失败 %s: %s" % (name, error), file=sys.stderr)
sys.exit(0 if count > 0 else 1)
PY
}

# 补装 wheelhouse 中缺失的少量小依赖（nvidia 运行时 + torch 纯 Python 依赖）
install_torch_small_deps() {
    local packages=(
        "nvidia-cuda-runtime-cu12==12.1.105"
        "nvidia-nvtx-cu12==12.1.105"
        "mpmath"
        "filelock"
        "jinja2"
        "fsspec"
    )
    info "补装 wheelhouse 缺少的小依赖：${packages[*]}"
    local pip_args=(install --disable-pip-version-check --no-input --no-deps --timeout 120 --retries 2)
    if [ -n "${PIC_POLISH_PYPI_INDEX:-}" ]; then
        pip_args+=(--index-url "$PIC_POLISH_PYPI_INDEX")
    fi
    if [ -d /tmp/nvwheels ]; then
        pip_args+=(--find-links /tmp/nvwheels)
    fi
    "$PY" -m pip "${pip_args[@]}" "${packages[@]}"
}

install_torch_online() {
    local indexes=("$TORCH_INDEX" "https://pypi.org/simple")
    local index attempt=0
    for index in "${indexes[@]}"; do
        attempt=$((attempt + 1))
        info "联网安装 PyTorch（第 $attempt 次，源: $index，约 2 GB，耗时取决于网络）..."
        if "$PY" -m pip install --disable-pip-version-check --no-input \
            --timeout 300 --retries 2 --index-url "$index" \
            "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"; then
            return 0
        fi
        warn "从 $index 安装失败"
        [ "$attempt" -lt "${#indexes[@]}" ] && sleep 3
    done
    return 1
}

install_torch() {
    if [ -n "${PIC_POLISH_SKIP_TORCH:-}" ] && [ "${PIC_POLISH_SKIP_TORCH}" != "0" ]; then
        warn 'PIC_POLISH_SKIP_TORCH 已设置，跳过 PyTorch 安装（服务将使用 pil 兼容引擎）'
        return 1
    fi
    if torch_import_ok; then
        ok "已安装 PyTorch：$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
        return 0
    fi

    local wh
    wh="$(find_wheelhouse || true)"
    if [ -n "$wh" ]; then
        info "检测到本地 wheelhouse: $wh"
        info '使用离线解包方式安装 PyTorch 全家桶（避免 pip 重新下载 2.9GB）...'
        if extract_wheels_from_wheelhouse "$wh"; then
            install_torch_small_deps || warn '小依赖补装失败（若 torch 仍可用可忽略）'
            if torch_import_ok; then
                if torch_cuda_ok; then
                    ok "PyTorch 离线安装成功（CUDA 可用）"
                else
                    warn 'PyTorch 可用但 CUDA 不可用（将以 CPU 推理或降级 pil）'
                fi
                return 0
            fi
            warn '离线解包后仍无法 import torch，改为联网安装...'
        else
            warn 'wheelhouse 解包失败，改为联网安装...'
        fi
    else
        info '未找到本地 wheelhouse，将联网安装 PyTorch（体积较大，请耐心等待）'
    fi

    if install_torch_online; then
        if torch_cuda_ok; then
            ok 'PyTorch 联网安装成功（CUDA 可用）'
        else
            warn 'PyTorch 安装成功但 CUDA 不可用（将以 CPU 推理或降级 pil）'
        fi
        return 0
    fi

    warn '自动安装 PyTorch 失败（网络/镜像问题）。'
    warn '服务仍可启动，但会降级为 pil 兼容引擎（无 AI 超分）。'
    info '可稍后重试本脚本，或手动安装后重跑：'
    info "  $PY -m pip install torch torchvision --index-url $TORCH_INDEX"
    info '离线场景：把 torch/torchvision/nvidia-* 轮子放入 wheelhouse 目录后重新运行。'
    return 1
}

# ---------------------------------------------------------------------------
# venv 创建（ensurepip 缺失时 virtualenv 兜底）
# ---------------------------------------------------------------------------
create_venv() {
    if ! command -v python3 >/dev/null 2>&1; then
        fail '未找到 python3，请先安装 Python 3.10+（Ubuntu: sudo apt install python3 python3-pip）。'
        return 1
    fi
    info "系统 Python: $(python3 -V 2>&1)"
    info "创建 venv: $VENV"
    if python3 -m venv "$VENV" 2>&1 | sed 's/^/    /'; then
        if [ -x "$PY" ]; then
            ok 'venv 创建完成'
            return 0
        fi
    fi
    warn 'python3 -m venv 失败（常见原因：缺少 ensurepip / python3-venv 包）。'
    info '兜底方案：用 pip --user --break-system-packages 安装 virtualenv 后重新创建。'
    rm -rf "$VENV"
    if python3 -m pip install --user --break-system-packages --disable-pip-version-check --no-input virtualenv; then
        info "使用 virtualenv 创建: $(python3 -m virtualenv --version 2>/dev/null || true)"
        if python3 -m virtualenv "$VENV"; then
            if [ -x "$PY" ]; then
                ok 'venv 创建完成（virtualenv）'
                return 0
            fi
        fi
    fi
    fail '创建 .venv-linux 失败。'
    info '可尝试手动安装系统 venv 支持后重试：'
    info '  sudo apt update && sudo apt install -y python3-venv python3-pip'
    info '  bash scripts/setup_env.sh'
    return 1
}

# ---------------------------------------------------------------------------
# 模型权重状态（只检查，不下载；下载由 start.sh 首次启动自动进行）
# ---------------------------------------------------------------------------
show_models_status() {
    [ -f "$FETCH_MODELS" ] || { warn "未找到 $FETCH_MODELS，跳过模型权重检查"; return 0; }
    local models_dir="${PIC_POLISH_MODELS_DIR:-$ROOT/models}"
    info "模型权重目录: $models_dir"
    local fetch_args=("$FETCH_MODELS" "--check")
    if [ -n "${PIC_POLISH_MODELS_DIR:-}" ]; then
        fetch_args+=("--models-dir" "$PIC_POLISH_MODELS_DIR")
    fi
    if "$PY" "${fetch_args[@]}" 2>&1 | sed 's/^/    /'; then
        ok '全部模型权重已就绪'
    else
        warn '部分模型权重缺失/不完整 —— 首次启动时 start.sh 会自动下载（也可手动）：'
        info "  $PY scripts/fetch_models.py"
    fi
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
title 'PicPolish Linux/WSL 环境准备'
info "项目根目录: $ROOT"
info "目标 venv:   $VENV"

if [ ! -f "$REQ" ]; then
    fail "缺少 $REQ，请确认项目完整。"
    exit 1
fi

WH="$(find_wheelhouse || true)"
if [ -n "$WH" ]; then
    info "本地 wheelhouse: $WH"
else
    info '本地 wheelhouse: 未找到（离线安装 torch 时将尝试 pip 缓存/联网）'
fi

if [ -d "$VENV" ]; then
    if [ ! -x "$PY" ]; then
        fail "目录存在但缺少可执行的 $PY，环境已损坏。"
        info '本脚本不会删除已有 venv；请手动删除后重试：'
        info "  rm -rf '$VENV' && bash scripts/setup_env.sh"
        exit 1
    fi
    info '检测到已存在的 .venv-linux（不会删除或重建），开始校验...'
    report="$(check_venv)"
    show_report "$report"

    # 1) Web 依赖
    if [ "$(report_missing_count "$report")" != "0" ]; then
        install_requirements "$WH" || true
        report="$(check_venv)"
    fi

    # 2) torch（缺失才安装）
    if [ -z "$(report_field "$report" torch)" ]; then
        title '安装 PyTorch（CUDA 12.1）'
        install_torch || true
        report="$(check_venv)"
    fi
else
    title '创建 .venv-linux'
    create_venv || exit 1

    title '安装 Web 依赖'
    install_requirements "$WH" || exit 1
    report="$(check_venv)"

    if [ -z "$(report_field "$report" torch)" ]; then
        title '安装 PyTorch（CUDA 12.1）'
        install_torch || true
        report="$(check_venv)"
    fi
fi

title '最终校验'
show_report "$report"
show_models_status

missing_count="$(report_missing_count "$report")"
if [ "$missing_count" != "0" ]; then
    fail 'Web 运行依赖不完整，服务无法启动；请检查上面的 pip 输出。'
    exit 1
fi

title '完成'
if [ -n "$(report_field "$report" torch)" ]; then
    if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        ok '环境准备完成：torch CUDA 可用（/api/health 应为 realesrgan/cuda）'
    else
        ok '环境准备完成：torch 可用但 CUDA 不可用（CPU 推理或降级 pil）'
    fi
else
    ok '环境准备完成：未安装 torch，将使用 pil 兼容引擎'
fi
info "venv Python: $PY"
info '启动服务: bash start.sh（首次启动会自动下载缺失的模型权重）'
exit 0
