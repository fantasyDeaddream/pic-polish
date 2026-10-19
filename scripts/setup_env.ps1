#Requires -Version 7.0
<#
.SYNOPSIS
    PicPolish 运行环境准备 / 修复脚本（Windows，幂等）。

.DESCRIPTION
    负责“解压后第一次启动”所需的全部 Python 环境准备，可重复运行：

      1. .venv 已存在 → 校验 Python / torch / CUDA / Web 依赖；
         缺 requirements 依赖（含 opencv-python-headless）→ 自动补装；
      2. 没有可用 torch → 自动 pip install torch：
           检测到 nvidia-smi → CUDA 12.1 轮子（cu121）
           否则                 → CPU 轮子（cpu）
         可用 PIC_POLISH_TORCH_INDEX 覆盖下载源；失败只警告并提示降级 pil，不阻塞；
      3. .venv 不存在 → 按候选 Python 顺序（PIC_POLISH_PYTHON → gnn → novel → openui
         → PATH python3.12/python/python3）优先选自带 torch 的，用 --system-site-packages
         建 venv；若没有任何候选自带 torch，则用第一个可用解释器建 venv 后自动安装 torch；
      4. 安装 requirements.txt（fastapi/uvicorn/python-multipart/pillow/numpy/opencv）；
      5. 校验并打印 torch / CUDA / Web 依赖 / 模型权重状态。

    本脚本**绝不删除**已存在的 .venv（损坏时提示手动删除），也**不会**重复下载已就绪的模型
    权重（模型下载由启动脚本在首次启动时自动调用 scripts/fetch_models.py 完成）。

.PARAMETER SkipModels
    跳过模型权重状态检查（更快）。

.EXAMPLE
    pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\setup_env.ps1

.NOTES
    退出码：0 = 环境可用（torch 安装失败但 Web 依赖完整时仍为 0，服务会降级 pil）；
            1 = 环境不可用（venv 损坏 / Web 依赖安装失败）。

    可用环境变量调整：
      PIC_POLISH_PYTHON         首选候选解释器
      PIC_POLISH_TORCH_INDEX    torch 安装源（默认按 CUDA 可用性选 cu121/cpu）
      PIC_POLISH_SKIP_TORCH=1   跳过自动安装 torch（只用现有环境）
      PIC_POLISH_MODELS_DIR     模型权重目录（默认 <项目>\models）
#>
[CmdletBinding()]
param(
    [switch]$SkipModels
)

$ErrorActionPreference = 'Stop'

try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch { }
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$VenvDir = Join-Path $Root '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$Requirements = Join-Path $Root 'requirements.txt'
$FetchModels = Join-Path $Root 'scripts\fetch_models.py'

function Write-Title([string]$Text) {
    Write-Host ''
    Write-Host "==== $Text ====" -ForegroundColor Cyan
}
function Write-Info([string]$Text) { Write-Host "  $Text" }
function Write-Ok([string]$Text) { Write-Host "  [OK] $Text" -ForegroundColor Green }
function Write-Warn([string]$Text) { Write-Host "  [警告] $Text" -ForegroundColor Yellow }
function Write-Fail([string]$Text) { Write-Host "  [错误] $Text" -ForegroundColor Red }

# requirements 包名 -> import 模块名（判断缺失用）
$script:RequirementModuleMap = @{
    'fastapi'                = 'fastapi'
    'uvicorn'                = 'uvicorn'
    'python-multipart'       = 'multipart'
    'pillow'                 = 'PIL'
    'numpy'                  = 'numpy'
    'opencv-python-headless' = 'cv2'
}

function Get-RequirementNames {
    if (-not (Test-Path -LiteralPath $Requirements)) { return @() }
    $names = @()
    foreach ($line in Get-Content -LiteralPath $Requirements -Encoding utf8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#')) { continue }
        $match = [regex]::Match($trimmed, '^([A-Za-z0-9][A-Za-z0-9._-]*)')
        if ($match.Success) { $names += $match.Groups[1].Value }
    }
    return $names
}

function Get-ModuleForRequirement([string]$Requirement) {
    if ($script:RequirementModuleMap.ContainsKey($Requirement)) {
        return $script:RequirementModuleMap[$Requirement]
    }
    return ($Requirement -replace '-', '_')
}

function Get-VenvReport([string]$PythonExe) {
    $code = @'
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
'@
    try {
        $raw = & $PythonExe -c $code 2>&1
        if ($LASTEXITCODE -ne 0) { return $null }
        $line = (($raw | Out-String).Trim() -split "`r?`n" | Where-Object { $_.Trim().StartsWith('{') } | Select-Object -Last 1)
        if (-not $line) { return $null }
        return $line | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Get-MissingRequirements([object]$Report) {
    $missing = @()
    foreach ($name in (Get-RequirementNames)) {
        $module = Get-ModuleForRequirement $name
        if ($Report.missing -contains $module) { $missing += $name }
    }
    return $missing
}

function Show-VenvReport([object]$Report) {
    Write-Info "Python:  $($Report.python)"
    if ($Report.torch) {
        Write-Ok "torch:   $($Report.torch)（CUDA 可用: $($Report.cuda)$(if ($Report.gpu) { "，GPU: $($Report.gpu)" })）"
    } else {
        Write-Warn 'torch:   未安装（服务会降级到 pil 兼容引擎）'
    }
    $missing = Get-MissingRequirements -Report $Report
    if ($missing.Count -gt 0) {
        Write-Warn "缺少依赖: $($missing -join ', ')"
    } else {
        Write-Ok 'Web 依赖: fastapi / uvicorn / python-multipart / pillow / numpy / opencv(cv2) 均可用'
    }
}

function Invoke-Pip([string]$PythonExe, [string[]]$Arguments) {
    & $PythonExe -m pip @Arguments 2>&1 | Out-Host
    return $LASTEXITCODE
}

# 补装 requirements.txt 缺失项；返回 $true 表示依赖已齐全。
function Install-PythonRequirements([string]$PythonExe, [object]$Report) {
    $missing = Get-MissingRequirements -Report $Report
    if ($missing.Count -eq 0) {
        Write-Ok 'requirements.txt 依赖已全部满足'
        return $true
    }
    Write-Info "补装缺失依赖: $($missing -join ', ')"
    Write-Info "命令: & '$PythonExe' -m pip install -r '$Requirements'"
    $code = Invoke-Pip -PythonExe $PythonExe -Arguments @(
        'install', '--disable-pip-version-check', '--no-input',
        '--timeout', '120', '--retries', '2', '-r', $Requirements
    )
    if ($code -ne 0) {
        Write-Fail "pip install 失败（退出码 $code）。"
        return $false
    }
    Write-Ok '依赖安装完成'
    return $true
}

function Test-NvidiaSmi {
    $cmd = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }
    $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }
    foreach ($path in @(
            'C:\Windows\System32\nvidia-smi.exe',
            'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'
        )) {
        if (Test-Path -LiteralPath $path) { return $path }
    }
    return $null
}

# 自动安装 torch；返回 $true 表示安装后可用。
function Install-Torch([string]$PythonExe) {
    if ($env:PIC_POLISH_SKIP_TORCH -and $env:PIC_POLISH_SKIP_TORCH -notin @('0', 'false', 'no')) {
        Write-Warn 'PIC_POLISH_SKIP_TORCH 已设置，跳过自动安装 torch（服务将使用 pil 兼容引擎）'
        return $false
    }
    $index = $env:PIC_POLISH_TORCH_INDEX
    if ($index) {
        Write-Info "torch 安装源（PIC_POLISH_TORCH_INDEX）: $index"
    } else {
        $smi = Test-NvidiaSmi
        if ($smi) {
            # 有 NVIDIA 显卡 → CUDA 12.1 轮子（与项目已验证的 torch 2.5.1+cu121 / WSL 一致）
            $index = 'https://download.pytorch.org/whl/cu121'
            Write-Info "检测到 NVIDIA 驱动（$smi）→ 安装 CUDA 12.1 版 torch"
        } else {
            $index = 'https://download.pytorch.org/whl/cpu'
            Write-Info '未检测到 nvidia-smi → 安装 CPU 版 torch'
        }
    }
    Write-Info '首次安装 torch 体积较大（约 2 GB），耗时取决于网络，请耐心等待...'
    $code = Invoke-Pip -PythonExe $PythonExe -Arguments @(
        'install', '--disable-pip-version-check', '--no-input',
        '--timeout', '120', '--retries', '2',
        'torch', 'torchvision', '--index-url', $index
    )
    if ($code -ne 0) {
        Write-Warn "自动安装 torch 失败（pip 退出码 $code，可能是网络/镜像问题）。"
        Write-Warn '服务仍可启动，但会降级为 pil 兼容引擎（无 AI 超分）。'
        Write-Info '可稍后重试本脚本，或按 https://pytorch.org 手动安装，例如：'
        Write-Info "  & '$PythonExe' -m pip install torch torchvision --index-url $index"
        if ($env:PIC_POLISH_TORCH_INDEX) {
            Write-Info '（PIC_POLISH_TORCH_INDEX 可指向内网镜像/本地 wheelhouse）'
        }
        return $false
    }
    $report = Get-VenvReport -PythonExe $PythonExe
    if ($report -and $report.torch) {
        Write-Ok "torch 安装成功: $($report.torch)（CUDA 可用: $($report.cuda)）"
        return $true
    }
    Write-Warn 'torch 安装命令已执行但无法 import，服务将降级 pil。'
    return $false
}

function Show-ModelStatus([string]$PythonExe) {
    if ($SkipModels) { return }
    if (-not (Test-Path -LiteralPath $FetchModels)) {
        Write-Warn "未找到 $FetchModels，跳过模型权重检查。"
        return
    }
    $modelsDir = if ($env:PIC_POLISH_MODELS_DIR) { $env:PIC_POLISH_MODELS_DIR } else { Join-Path $Root 'models' }
    Write-Info "模型权重目录: $modelsDir"
    $fetchArgs = @($FetchModels, '--check')
    if ($env:PIC_POLISH_MODELS_DIR) { $fetchArgs += @('--models-dir', $env:PIC_POLISH_MODELS_DIR) }
    $raw = & $PythonExe @fetchArgs 2>&1
    $code = $LASTEXITCODE
    foreach ($line in (($raw | Out-String).Trim() -split "`r?`n")) {
        if ($line -ne '') { Write-Info "  $line" }
    }
    if ($code -eq 0) {
        Write-Ok '全部模型权重已就绪'
    } else {
        Write-Warn '部分模型权重缺失/不完整 —— 首次启动时 start.bat / server.ps1 会自动下载（也可手动运行）：'
        Write-Info "  & '$PythonExe' scripts\fetch_models.py"
    }
}

function Get-CandidatePythons {
    $list = [System.Collections.Generic.List[string]]::new()
    if ($env:PIC_POLISH_PYTHON) { $list.Add($env:PIC_POLISH_PYTHON) }
    $list.Add('E:\ProgramData\anaconda3new\envs\gnn\python.exe')
    $list.Add('E:\ProgramData\anaconda3new\envs\novel\python.exe')
    $list.Add('E:\ProgramData\anaconda3new\envs\openui\python.exe')
    foreach ($name in @('python3.12', 'python', 'python3')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $cmd -or -not $cmd.Source) { continue }
        if ($cmd.Source -like '*\WindowsApps\*') { continue }  # Windows 商店占位符
        $list.Add($cmd.Source)
    }
    # 去重并保持顺序
    $seen = @{}
    $result = @()
    foreach ($item in $list) {
        if (-not $item) { continue }
        $key = $item.ToLowerInvariant()
        if ($seen.ContainsKey($key)) { continue }
        $seen[$key] = $true
        $result += $item
    }
    return $result
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
Write-Title 'PicPolish 环境准备（Windows）'
Write-Info "项目根目录: $Root"
Write-Info "目标 venv:   $VenvDir"

if (-not (Test-Path -LiteralPath $Requirements)) {
    Write-Fail "缺少 $Requirements，无法准备依赖。"
    exit 1
}

# ===========================================================================
# 分支 A：.venv 已存在 → 校验 + 补装
# ===========================================================================
if (Test-Path -LiteralPath $VenvDir) {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Write-Fail ".venv 目录存在但缺少 .venv\Scripts\python.exe，环境已损坏。"
        Write-Info '本脚本不会自动删除已有 .venv；请手动删除后重试：'
        Write-Info "  Remove-Item -Recurse -Force '$VenvDir'"
        Write-Info '  pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\setup_env.ps1'
        exit 1
    }
    Write-Info '检测到已存在的 .venv（不会删除或重建），开始校验...'
    $report = Get-VenvReport -PythonExe $VenvPython
    if (-not $report) {
        Write-Fail "$VenvPython 无法运行，环境可能已损坏。"
        exit 1
    }
    Show-VenvReport -Report $report

    $reqOk = Install-PythonRequirements -PythonExe $VenvPython -Report $report
    if (-not $reqOk) {
        Write-Fail 'Web 运行依赖安装失败，服务无法启动；请检查网络/代理后重试本脚本。'
        exit 1
    }

    $report = Get-VenvReport -PythonExe $VenvPython
    if ($report -and -not $report.torch) {
        Install-Torch -PythonExe $VenvPython | Out-Null
        $report = Get-VenvReport -PythonExe $VenvPython
    }

    Write-Title '最终校验'
    Show-VenvReport -Report $report
    Show-ModelStatus -PythonExe $VenvPython
    $missing = Get-MissingRequirements -Report $report
    if ($missing.Count -gt 0) {
        Write-Fail "依赖仍不完整：$($missing -join ', ')"
        exit 1
    }
    Write-Title '完成'
    Write-Ok '环境可用（幂等：已存在的 venv 与已就绪的权重不会被重建/重复下载）'
    Write-Info '启动服务: 双击 start.bat，或 pwsh -File scripts\server.ps1 -Action start'
    exit 0
}

# ===========================================================================
# 分支 B：.venv 不存在 → 选择解释器并创建
# ===========================================================================
Write-Title '寻找可用的 Python 解释器'
Write-Info '优先选择自带 torch 的现有环境；若都没有，将自动安装 torch。'

$candidates = Get-CandidatePythons
$selected = $null
$fallback = $null
foreach ($candidate in $candidates) {
    if (-not (Test-Path -LiteralPath $candidate)) {
        Write-Info "跳过（不存在）: $candidate"
        continue
    }
    try {
        $probe = & $candidate -c "import sys, torch; print(sys.version.split()[0], torch.__version__)" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $line = (($probe | Out-String).Trim() -split "`r?`n" | Select-Object -Last 1)
            Write-Ok "选用: $candidate（自带 torch $line）"
            $selected = $candidate
            break
        }
        if (-not $fallback) {
            $ver = & $candidate -c "import sys; print(sys.version.split()[0])" 2>&1
            $fallback = $candidate
            Write-Info "候选（无 torch，可自动安装）: $candidate（Python $((($ver | Out-String).Trim()))）"
        }
    } catch {
        Write-Info "跳过（无法运行）: $candidate"
    }
}

if (-not $selected) {
    if (-not $fallback) {
        Write-Fail '没有找到可用的 Python 解释器，无法创建 .venv。'
        Write-Info '请安装 Python 3.10-3.12（勾选 Add to PATH）后重试，或通过环境变量指定：'
        Write-Info '  $env:PIC_POLISH_PYTHON = "D:\path\to\python.exe"; pwsh -File scripts\setup_env.ps1'
        exit 1
    }
    $selected = $fallback
    Write-Warn "没有候选环境自带 torch，将用 $selected 创建 venv 并自动安装 torch（约 2 GB）。"
}

Write-Title '创建项目 venv（--system-site-packages）'
Write-Info "解释器: $selected"
Write-Info "目标:   $VenvDir"
& $selected -m venv --system-site-packages $VenvDir 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
    Write-Fail "创建 venv 失败（退出码 $LASTEXITCODE）。"
    Write-Info "  & '$selected' -m venv --system-site-packages '$VenvDir'"
    exit 1
}
Write-Ok 'venv 创建完成'

Write-Title '安装 Web 运行依赖'
$report = Get-VenvReport -PythonExe $VenvPython
$reqOk = Install-PythonRequirements -PythonExe $VenvPython -Report $report
if (-not $reqOk) {
    Write-Fail 'Web 运行依赖安装失败；请检查网络/代理后重新运行本脚本。'
    exit 1
}

$report = Get-VenvReport -PythonExe $VenvPython
if ($report -and -not $report.torch) {
    Write-Title '安装 PyTorch'
    Install-Torch -PythonExe $VenvPython | Out-Null
    $report = Get-VenvReport -PythonExe $VenvPython
}

Write-Title '最终校验'
Show-VenvReport -Report $report
Show-ModelStatus -PythonExe $VenvPython
$missing = Get-MissingRequirements -Report $report
if ($missing.Count -gt 0) {
    Write-Fail "依赖不完整：$($missing -join ', ')"
    exit 1
}

Write-Title '完成'
Write-Ok '环境准备完成'
Write-Info "venv Python: $VenvPython"
Write-Info '启动服务: 双击 start.bat（首次启动会自动下载缺失的模型权重）'
exit 0
