#!/usr/bin/env bash
# =============================================================================
#  PicPolish 停止脚本（Linux / WSL Ubuntu）
#
#  用法:
#      bash stop.sh
#
#  行为:
#    * 读取 .runtime/server.json，校验 PID 命令行后终止服务（防误杀）
#    * 兜底扫描监听端口，终止本项目的残留实例
#    * 清空 cache/uploads、cache/results、cache/tmp 内所有文件（保留目录）
#    * 删除 .runtime/server.json；未运行时也执行清理并返回 0（幂等）
#
#  实际逻辑在 scripts/server.sh，本文件只是薄包装（参数原样透传）。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/scripts/server.sh" stop "$@"
