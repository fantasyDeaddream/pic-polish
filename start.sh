#!/usr/bin/env bash
# =============================================================================
#  PicPolish 启动脚本（Linux / WSL Ubuntu）
#
#  用法:
#      bash start.sh                          # 默认 http://127.0.0.1:8765
#      bash start.sh --port 8798 --no-browser
#      PIC_POLISH_ENGINE=pil bash start.sh    # WSL 无 torch 时强制兼容引擎
#
#  实际逻辑在 scripts/server.sh，本文件只是薄包装（参数原样透传）。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/scripts/server.sh" start "$@"
