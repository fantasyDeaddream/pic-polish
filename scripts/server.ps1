#Requires -Version 7.0
<#
.SYNOPSIS
    PicPolish 服务管理脚本（Windows / PowerShell 7+）。

.DESCRIPTION
    统一管理 PicPolish 本地服务的启动、停止、状态、日志与重启。
    根目录的 start.bat / stop.bat 是双击友好的薄包装，最终都调用本脚本。

    约定（docs/CONTRACT.md §4、docs/API.md §12）：
      * Python 固定使用项目 venv：.venv\Scripts\python.exe
      * 启动命令固定为: python -m uvicorn app.main:app --host <host> --port <port>
      * 后台进程 stdout/stderr 重定向到 logs\server.log / logs\server.err.log
      * 运行信息写入 .runtime\server.json（{pid,host,port,url,started_at,engine,log}）
      * stop 会校验 PID 命令行（防 PID 复用误杀），并清空 cache 下 uploads/results/tmp
        内所有文件（保留目录），未运行时也执行清理并返回 0（幂等）。

    分发版“首次启动自动配置”（start 时自动完成，无需手动准备）：
      * .venv 缺失 → 自动运行 scripts\setup_env.ps1（准备 Python/torch/依赖）；
      * 模型权重缺失/不完整 → 自动运行 scripts\fetch_models.py 下载（默认最长 20 分钟，
        超时/失败只警告并继续启动，引擎会自动降级）；
      * 可用 PIC_POLISH_AUTO_SETUP=0 / PIC_POLISH_AUTO_DOWNLOAD=0 关闭自动行为，
        PIC_POLISH_MODEL_DOWNLOAD_TIMEOUT 调整下载超时秒数。

.PARAMETER Action
    start | stop | status | restart | logs，默认 start。

.PARAMETER Port
    监听端口。默认取 PIC_POLISH_PORT，否则 8765；被占用时自动向后顺延。

.PARAMETER Host
    监听地址（注意：PowerShell 的 $Host 是只读自动变量，故内部名称为 BindHost，
    命令行仍使用 -Host 或 -BindHost）。默认取 PIC_POLISH_HOST，否则 127.0.0.1。

.PARAMETER NoBrowser
    启动成功后不自动打开浏览器。

.PARAMETER Foreground
    调试用：在前台直接运行 uvicorn（Ctrl+C 结束），不写 runtime 文件。

.PARAMETER Lines
    -Action logs 时显示的日志尾行数，默认 40。

.EXAMPLE
    pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\server.ps1 -Action start -NoBrowser -Port 8795

.EXAMPLE
    pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\server.ps1 -Action stop

.NOTES
    退出码：0 成功（含“本来就没运行”的 stop / 幂等 start）；1 失败。
#>
[CmdletBinding()]
param(
    [ValidateSet('start', 'stop', 'status', 'restart', 'logs')]
    [string]$Action = 'start',

    # 注意：$Host 是 PowerShell 只读自动变量，参数变量必须叫别的名字，
    # 再通过 Alias 让命令行仍然可以使用 -Host。
    [Alias('Host')]
    [string]$BindHost = '',

    [int]$Port = 0,

    [switch]$NoBrowser,

    [switch]$Foreground,

    [int]$Lines = 40
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# 控制台 / 子进程环境（UTF-8，避免 Windows 默认代码页导致日志乱码）
# ---------------------------------------------------------------------------
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch { }
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$VenvPythonDefault = Join-Path $Root '.venv\Scripts\python.exe'
$Python = if ($env:PIC_POLISH_VENV_PYTHON) { $env:PIC_POLISH_VENV_PYTHON } else { $VenvPythonDefault }

$CacheDir = Join-Path $Root 'cache'
$UploadDir = Join-Path $CacheDir 'uploads'
$ResultDir = Join-Path $CacheDir 'results'
$TmpDir = Join-Path $CacheDir 'tmp'
$LogDir = Join-Path $Root 'logs'
$OutLog = Join-Path $LogDir 'server.log'
$ErrLog = Join-Path $LogDir 'server.err.log'
$RuntimeDir = Join-Path $Root '.runtime'
$RuntimeFile = Join-Path $RuntimeDir 'server.json'

# 首次启动自动配置相关
$ModelsDir = if ($env:PIC_POLISH_MODELS_DIR) { $env:PIC_POLISH_MODELS_DIR } else { Join-Path $Root 'models' }
$SetupScript = Join-Path $Root 'scripts\setup_env.ps1'
$FetchModelsScript = Join-Path $Root 'scripts\fetch_models.py'
$ModelDownloadLog = Join-Path $LogDir 'models_download.log'
$ModelDownloadErrLog = Join-Path $LogDir 'models_download.err.log'
$ModelDownloadTimeout = 1200
if ($env:PIC_POLISH_MODEL_DOWNLOAD_TIMEOUT) {
    $parsedTimeout = 0
    if ([int]::TryParse($env:PIC_POLISH_MODEL_DOWNLOAD_TIMEOUT, [ref]$parsedTimeout) -and $parsedTimeout -gt 0) {
        $ModelDownloadTimeout = $parsedTimeout
    }
}

$DefaultHost = if ($env:PIC_POLISH_HOST) { $env:PIC_POLISH_HOST } else { '127.0.0.1' }
$DefaultPort = 8765
if ($env:PIC_POLISH_PORT) {
    $parsed = 0
    if ([int]::TryParse($env:PIC_POLISH_PORT, [ref]$parsed) -and $parsed -gt 0) { $DefaultPort = $parsed }
}
$BindHost = if ($BindHost) { $BindHost } else { $DefaultHost }
$RequestedPort = if ($Port -gt 0) { $Port } else { $DefaultPort }
$EngineEnv = if ($env:PIC_POLISH_ENGINE) { $env:PIC_POLISH_ENGINE } else { 'auto' }

$HealthUrl = $null
$ProbeHost = if ($BindHost -in @('0.0.0.0', '::', '*')) { '127.0.0.1' } else { $BindHost }

# ---------------------------------------------------------------------------
# 输出辅助
# ---------------------------------------------------------------------------
function Write-Title([string]$Text) {
    Write-Host ''
    Write-Host "==== $Text ====" -ForegroundColor Cyan
}
function Write-Info([string]$Text) { Write-Host "  $Text" }
function Write-Ok([string]$Text) { Write-Host "  [OK] $Text" -ForegroundColor Green }
function Write-Warn([string]$Text) { Write-Host "  [警告] $Text" -ForegroundColor Yellow }
function Write-Fail([string]$Text) { Write-Host "  [错误] $Text" -ForegroundColor Red }

function Format-Bytes([long]$Bytes) {
    if ($Bytes -ge 1GB) { return ('{0:N2} GB' -f ($Bytes / 1GB)) }
    if ($Bytes -ge 1MB) { return ('{0:N2} MB' -f ($Bytes / 1MB)) }
    if ($Bytes -ge 1KB) { return ('{0:N1} KB' -f ($Bytes / 1KB)) }
    return "$Bytes B"
}

# ---------------------------------------------------------------------------
# 通用探测函数
# ---------------------------------------------------------------------------
function Ensure-RuntimeDirs {
    foreach ($dir in @($CacheDir, $UploadDir, $ResultDir, $TmpDir, $LogDir, $RuntimeDir)) {
        if (-not (Test-Path -LiteralPath $dir)) {
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
        }
    }
}

function Read-Runtime {
    if (-not (Test-Path -LiteralPath $RuntimeFile)) { return $null }
    try {
        $raw = Get-Content -LiteralPath $RuntimeFile -Raw -Encoding utf8
        if (-not $raw.Trim()) { return $null }
        return $raw | ConvertFrom-Json
    } catch {
        Write-Warn "无法解析 .runtime\server.json（将忽略）：$($_.Exception.Message)"
        return $null
    }
}

function Write-Runtime([hashtable]$Data) {
    Ensure-RuntimeDirs
    $json = $Data | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($RuntimeFile, $json, [System.Text.UTF8Encoding]::new($false))
}

function Remove-Runtime {
    if (Test-Path -LiteralPath $RuntimeFile) {
        Remove-Item -LiteralPath $RuntimeFile -Force -ErrorAction SilentlyContinue
        return $true
    }
    return $false
}

# 进程快照缓存：停止流程要扫描全部监听端口，逐个 PID 查 WMI 会非常慢（可达 20s+），
# 因此这里一次性取全量进程表（pid -> 父 pid / 命令行 / 映像名），后续判断全是内存查询。
$script:ProcessTable = $null

function Update-ProcessTable {
    $table = @{}
    try {
        foreach ($proc in (Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)) {
            $table[[int]$proc.ProcessId] = [pscustomobject]@{
                ParentProcessId = [int]$proc.ParentProcessId
                CommandLine     = if ($proc.CommandLine) { [string]$proc.CommandLine } else { '' }
                Name            = [string]$proc.Name
            }
        }
    } catch { }
    $script:ProcessTable = $table
}

function Get-ProcessEntry([int]$ProcessId) {
    if ($null -ne $script:ProcessTable) {
        return $script:ProcessTable[$ProcessId]
    }
    try {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction Stop
        if ($proc) {
            return [pscustomobject]@{
                ParentProcessId = [int]$proc.ParentProcessId
                CommandLine     = if ($proc.CommandLine) { [string]$proc.CommandLine } else { '' }
                Name            = [string]$proc.Name
            }
        }
    } catch { }
    return $null
}

function Get-ProcessCommandLine([int]$ProcessId) {
    $entry = Get-ProcessEntry -ProcessId $ProcessId
    if ($entry -and $entry.CommandLine) { return [string]$entry.CommandLine }
    return $null
}

# 校验 PID 确实是“本项目的 uvicorn 进程”，避免 PID 复用误杀。
# 判定条件：命令行含 uvicorn，且包含项目根目录（正常情况：命令行里是
# <项目>\.venv\Scripts\python.exe）或显式指定的 Python 解释器路径。
# 说明：conda 环境导出的 venv 里 python.exe 可能是“转发器”，真正监听端口的是
# 它的子进程（基础 conda python），因此调用方还应配合祖先链解析一起判断。
function Test-OwnProcess([int]$ProcessId) {
    if ($ProcessId -le 0) { return $false }
    $entry = Get-ProcessEntry -ProcessId $ProcessId
    if (-not $entry) { return $false }
    $cmd = [string]$entry.CommandLine
    if (-not $cmd) { return $false }
    $hasRoot = $cmd.IndexOf($Root, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    $hasPython = $cmd.IndexOf($Python, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    $hasUvicorn = $cmd.IndexOf('uvicorn', [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    return ($hasUvicorn -and ($hasRoot -or $hasPython))
}

# 从给定 PID 开始向上检查祖先链（最多 4 层），返回第一个被校验为本项目 uvicorn
# 的 PID；找不到返回 0。用于兜底：监听端口的可能是 venv python.exe 的子进程。
function Resolve-OwnProcessPid([int]$ProcessId) {
    $cursor = $ProcessId
    for ($depth = 0; $depth -lt 4; $depth++) {
        if ($cursor -le 0) { return 0 }
        if (Test-OwnProcess -ProcessId $cursor) { return $cursor }
        $entry = Get-ProcessEntry -ProcessId $cursor
        if (-not $entry) { return 0 }
        $cursor = [int]$entry.ParentProcessId
    }
    return 0
}

function Test-PortFree([int]$PortNumber) {
    $listen = Get-NetTCPConnection -State Listen -LocalPort $PortNumber -ErrorAction SilentlyContinue
    if ($listen) { return $false }
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $PortNumber)
        $listener.Server.ExclusiveAddressUse = $true
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener) { try { $listener.Stop() } catch { } }
    }
}

# 端口上若已有“本项目的 uvicorn”监听，返回对应（已校验的）PID，否则返回 0。
function Get-OwnListenerPid([int]$PortNumber) {
    $conns = Get-NetTCPConnection -State Listen -LocalPort $PortNumber -ErrorAction SilentlyContinue
    if (-not $conns) { return 0 }
    foreach ($conn in $conns) {
        $ownerPid = [int]$conn.OwningProcess
        $ownPid = Resolve-OwnProcessPid -ProcessId $ownerPid
        if ($ownPid -gt 0) { return $ownPid }
    }
    return 0
}

function Find-FreePort([int]$StartPort, [int]$MaxTries = 100) {
    for ($i = 0; $i -lt $MaxTries; $i++) {
        $candidate = $StartPort + $i
        if ($candidate -gt 65535) { break }
        if (Test-PortFree -PortNumber $candidate) { return $candidate }
    }
    return 0
}

function Get-Health([string]$BaseUrl, [int]$TimeoutSec = 2) {
    try {
        $resp = Invoke-RestMethod -Uri "$BaseUrl/api/health" -TimeoutSec $TimeoutSec -ErrorAction Stop
        return $resp
    } catch {
        return $null
    }
}

# 轮询 /api/health，直到成功 / 超时 / 进程退出。成功返回 health 对象，失败返回 $null。
function Wait-Health([string]$BaseUrl, [int]$TimeoutSeconds, [int]$WatchPid = 0) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $health = Get-Health -BaseUrl $BaseUrl
        if ($health) { return $health }
        if ($WatchPid -gt 0 -and -not (Get-Process -Id $WatchPid -ErrorAction SilentlyContinue)) {
            return $null
        }
        Start-Sleep -Milliseconds 500
    }
    return $null
}

function Format-Engine([object]$Engine) {
    if ($null -eq $Engine) { return '未知（服务未返回引擎信息）' }
    $backend = if ($Engine.backend) { $Engine.backend } else { '?' }
    if ($backend -eq 'realesrgan') {
        $parts = @('AI 超分', "device=$($Engine.device)")
        if ($Engine.gpu) { $parts += "GPU=$($Engine.gpu)" }
        if ($Engine.model) { $parts += "model=$($Engine.model)" }
        return ($parts -join ' / ')
    }
    return "兼容模式（pil: Lanczos + 轻锐化）（backend=$backend）"
}

function Get-CacheStats {
    $count = 0
    $bytes = [long]0
    foreach ($dir in @($UploadDir, $ResultDir, $TmpDir)) {
        if (-not (Test-Path -LiteralPath $dir)) { continue }
        $items = Get-ChildItem -LiteralPath $dir -Force -Recurse -File -ErrorAction SilentlyContinue
        foreach ($item in $items) {
            $count++
            try { $bytes += $item.Length } catch { }
        }
    }
    return [pscustomobject]@{ Count = $count; Bytes = $bytes }
}

function Show-LogTail([string]$Path, [int]$TailLines) {
    if (Test-Path -LiteralPath $Path) {
        Write-Host "--- $Path（最后 $TailLines 行）---" -ForegroundColor DarkGray
        Get-Content -LiteralPath $Path -Tail $TailLines -Encoding utf8 -ErrorAction SilentlyContinue
    } else {
        Write-Host "--- $Path（不存在）---" -ForegroundColor DarkGray
    }
}

function Get-ListenerPids {
    $pids = @()
    $conns = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue
    if ($conns) {
        foreach ($conn in $conns) {
            $p = [int]$conn.OwningProcess
            if ($p -gt 0 -and $pids -notcontains $p) { $pids += $p }
        }
    }
    return $pids
}

# 杀掉一个“已校验为我们的”进程树。
function Stop-OwnProcessTree([int]$ProcessId, [string]$Reason) {
    Write-Info "终止进程 PID=$ProcessId（$Reason）..."
    $output = & taskkill.exe /PID $ProcessId /T /F 2>&1
    Write-Host ('    ' + (($output | Out-String).Trim())) -ForegroundColor DarkGray
    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { return $true }
        Start-Sleep -Milliseconds 300
    }
    try { Stop-Process -Id $ProcessId -Force -ErrorAction Stop } catch { }
    Start-Sleep -Milliseconds 500
    return -not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

# 清空 cache\uploads|results|tmp 下所有文件（保留三个目录本身）。
function Clear-ImageCache {
    Ensure-RuntimeDirs
    $removed = 0
    $freed = [long]0
    $failed = 0
    foreach ($dir in @($UploadDir, $ResultDir, $TmpDir)) {
        # 1) 先删文件（按文件计数、按大小统计释放字节）
        $files = Get-ChildItem -LiteralPath $dir -Force -Recurse -File -ErrorAction SilentlyContinue
        foreach ($file in $files) {
            $size = 0
            try { $size = $file.Length } catch { }
            try {
                Remove-Item -LiteralPath $file.FullName -Force -ErrorAction Stop
                $removed++
                $freed += $size
            } catch {
                # 可能被杀进程短暂占用，稍等重试一次
                Start-Sleep -Milliseconds 300
                try {
                    Remove-Item -LiteralPath $file.FullName -Force -ErrorAction Stop
                    $removed++
                    $freed += $size
                } catch {
                    $failed++
                }
            }
        }
        # 2) 再删除残留子目录（从深到浅）
        $dirs = Get-ChildItem -LiteralPath $dir -Force -Recurse -Directory -ErrorAction SilentlyContinue |
            Sort-Object { $_.FullName.Length } -Descending
        foreach ($sub in $dirs) {
            try { Remove-Item -LiteralPath $sub.FullName -Force -Recurse -ErrorAction Stop } catch { $failed++ }
        }
    }
    return [pscustomobject]@{ RemovedFiles = $removed; FreedBytes = $freed; Failed = $failed }
}

# ---------------------------------------------------------------------------
# 首次启动自动配置：环境准备 + 模型权重下载
# ---------------------------------------------------------------------------
function Test-AutoDisabled([string]$Value) {
    if (-not $Value) { return $false }
    return ($Value.Trim().ToLowerInvariant() -in @('0', 'false', 'no', 'off'))
}

function Get-PwshPath {
    $cmd = Get-Command pwsh -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }
    try { return (Get-Process -Id $PID).Path } catch { return 'pwsh' }
}

# 运行 scripts\setup_env.ps1（用子进程调用：setup 脚本里的 exit 不会终止本进程）
function Invoke-AutoSetup {
    if (Test-AutoDisabled $env:PIC_POLISH_AUTO_SETUP) {
        Write-Info '已禁用自动环境准备（PIC_POLISH_AUTO_SETUP=0）。'
        return 1
    }
    if (-not (Test-Path -LiteralPath $SetupScript)) {
        Write-Fail "未找到 $SetupScript，无法自动准备环境。"
        return 1
    }
    Write-Title '首次启动：自动准备运行环境'
    Write-Info '未检测到可用的 .venv，正在自动运行 scripts\setup_env.ps1（首次可能需要几分钟）...'
    Write-Info '可用 PIC_POLISH_AUTO_SETUP=0 关闭该行为。'
    $pwshPath = Get-PwshPath
    & $pwshPath -NoLogo -NoProfile -ExecutionPolicy Bypass -File $SetupScript
    return $LASTEXITCODE
}

# 模型权重是否全部就绪（调用 fetch_models.py --check；只检查文件，不加载模型）
function Test-ModelsReady {
    if (-not (Test-Path -LiteralPath $FetchModelsScript)) { return $true }
    $fetchArgs = @($FetchModelsScript, '--check')
    if ($env:PIC_POLISH_MODELS_DIR) { $fetchArgs += @('--models-dir', $env:PIC_POLISH_MODELS_DIR) }
    $null = & $Python @fetchArgs 2>&1
    return ($LASTEXITCODE -eq 0)
}

# 自动下载缺失的模型权重（最长 $ModelDownloadTimeout 秒；失败/超时只警告，不阻塞启动）
function Invoke-ModelDownload {
    if (Test-AutoDisabled $env:PIC_POLISH_AUTO_DOWNLOAD) {
        Write-Info '已禁用自动下载模型权重（PIC_POLISH_AUTO_DOWNLOAD=0）。'
        return $false
    }
    if (-not (Test-Path -LiteralPath $FetchModelsScript)) {
        Write-Warn "未找到 $FetchModelsScript，跳过模型权重下载。"
        return $false
    }
    Write-Title '首次启动：自动下载模型权重'
    Write-Info "权重目录: $ModelsDir"
    Write-Info "下载日志: $ModelDownloadLog"
    Write-Info "最长等待 $([math]::Round($ModelDownloadTimeout / 60, 0)) 分钟（PIC_POLISH_MODEL_DOWNLOAD_TIMEOUT 可调，PIC_POLISH_AUTO_DOWNLOAD=0 关闭自动下载）"
    Ensure-RuntimeDirs
    Remove-Item -LiteralPath $ModelDownloadLog, $ModelDownloadErrLog -Force -ErrorAction SilentlyContinue
    $fetchArgs = @($FetchModelsScript)
    if ($env:PIC_POLISH_MODELS_DIR) { $fetchArgs += @('--models-dir', $env:PIC_POLISH_MODELS_DIR) }
    try {
        $proc = Start-Process -FilePath $Python -ArgumentList $fetchArgs -WorkingDirectory $Root `
            -RedirectStandardOutput $ModelDownloadLog -RedirectStandardError $ModelDownloadErrLog `
            -WindowStyle Hidden -PassThru
    } catch {
        Write-Warn "启动模型下载进程失败：$($_.Exception.Message)"
        Write-Info "可手动运行: & '$Python' scripts\fetch_models.py"
        return $false
    }
    $startedAt = Get-Date
    $lastReport = -15
    while (-not $proc.HasExited) {
        $elapsed = [int]((Get-Date) - $startedAt).TotalSeconds
        if ($elapsed -ge $ModelDownloadTimeout) {
            Write-Warn "模型下载超过 $ModelDownloadTimeout 秒，终止下载进程（已下载内容会保留为 .part，下次重新下载）。"
            try { & taskkill.exe /PID $proc.Id /T /F 2>&1 | Out-Null } catch { }
            Write-Warn '服务将继续启动；缺失的模型会自动降级（可稍后手动重试下载）。'
            Write-Info "可手动运行: & '$Python' scripts\fetch_models.py"
            return $false
        }
        if (($elapsed - $lastReport) -ge 10) {
            $lastReport = $elapsed
            $tail = ''
            if (Test-Path -LiteralPath $ModelDownloadLog) {
                $tail = [string](Get-Content -LiteralPath $ModelDownloadLog -Tail 1 -Encoding utf8 -ErrorAction SilentlyContinue)
            }
            if ($tail) {
                if ($tail.Length -gt 110) { $tail = $tail.Substring(0, 110) }
                Write-Info "[$elapsed s] $($tail.Trim())"
            } else {
                Write-Info "[$elapsed s] 下载中..."
            }
        }
        Start-Sleep -Milliseconds 500
    }
    $code = $proc.ExitCode
    if ($code -eq 0) {
        Write-Ok '模型权重下载完成'
        return $true
    }
    Write-Warn "模型权重下载失败（退出码 $code）。"
    Show-LogTail -Path $ModelDownloadErrLog -TailLines 10
    Show-LogTail -Path $ModelDownloadLog -TailLines 12
    Write-Warn '服务将继续启动；缺失的模型会自动降级（可稍后手动重试下载）。'
    Write-Info "可手动运行: & '$Python' scripts\fetch_models.py"
    return $false
}

# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------
function Invoke-Start {
    Ensure-RuntimeDirs
    Write-Title 'PicPolish 启动'
    Write-Info "项目根目录: $Root"
    Write-Info "Python:      $Python"
    Write-Info "引擎模式:    $EngineEnv（auto=自动选择 realesrgan/pil）"

    # 0) 分发版核心：首次启动自动准备 Python 环境
    if (-not (Test-Path -LiteralPath $Python)) {
        $setupCode = Invoke-AutoSetup
        if ($setupCode -ne 0 -or -not (Test-Path -LiteralPath $Python)) {
            Write-Fail "自动环境准备失败（退出码 $setupCode），未找到 Python 解释器：$Python"
            Write-Info '可手动运行: pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\setup_env.ps1'
            return 1
        }
        Write-Ok '运行环境已自动准备完成'
    }

    # 1) 幂等：runtime 文件 + 进程校验 + 健康检查都通过 → 直接返回现有实例
    $runtime = Read-Runtime
    if ($runtime -and $runtime.pid) {
        $existingPid = [int]$runtime.pid
        if (Test-OwnProcess -ProcessId $existingPid) {
            $existingUrl = if ($runtime.url) { [string]$runtime.url } else { "http://${ProbeHost}:$($runtime.port)" }
            Write-Info "检测到已在运行（PID=$existingPid），等待健康检查..."
            $health = Wait-Health -BaseUrl $existingUrl -TimeoutSeconds 30 -WatchPid $existingPid
            if ($health) {
                Write-Ok 'PicPolish 已在运行（本次 start 为幂等调用，未重复启动）'
                Write-Info "URL:   $existingUrl"
                Write-Info "PID:   $existingPid"
                Write-Info "引擎:  $(Format-Engine -Engine $health.engine)"
                if ($runtime.log -and $runtime.log.stdout) { Write-Info "日志:  $(Join-Path $Root ([string]$runtime.log.stdout))" }
                return 0
            }
            Write-Warn "已有进程 PID=$existingPid 存活但健康检查失败，请用 -Action logs 查看日志，或先 -Action stop 再启动。"
            return 1
        }
        Write-Warn "发现陈旧的运行记录（PID=$($runtime.pid) 已不是本项目的 uvicorn 进程），清理后继续启动。"
        Remove-Runtime | Out-Null
    }

    # 1.5) 分发版核心：首次启动自动下载缺失的模型权重（失败/超时只警告，不阻塞启动）
    if (Test-ModelsReady) {
        Write-Info "模型权重:   已就绪（$ModelsDir）"
    } else {
        Write-Warn "检测到模型权重缺失/不完整（$ModelsDir），尝试自动下载..."
        Invoke-ModelDownload | Out-Null
        if (Test-ModelsReady) {
            Write-Ok '模型权重已就绪'
        } else {
            Write-Warn '仍有模型权重缺失；对应模型将由引擎自动降级。可稍后手动重试下载。'
        }
    }

    # 2) 选端口：显式/默认端口被占用时自动向后顺延；若占用者正是本项目的实例则直接汇报
    $actualPort = $RequestedPort
    if (-not (Test-PortFree -PortNumber $actualPort)) {
        $ownPid = Get-OwnListenerPid -PortNumber $actualPort
        if ($ownPid -gt 0) {
            $url = "http://${ProbeHost}:$actualPort"
            $health = Wait-Health -BaseUrl $url -TimeoutSeconds 30 -WatchPid $ownPid
            Write-Ok "端口 $actualPort 上已有本项目实例在运行（PID=$ownPid），未重复启动。"
            Write-Info "URL:   $url"
            Write-Info "PID:   $ownPid"
            if ($health) { Write-Info "引擎:  $(Format-Engine -Engine $health.engine)" }
            return 0
        }
        $freePort = Find-FreePort -StartPort ($actualPort + 1)
        if ($freePort -le 0) {
            Write-Fail "端口 $actualPort 被占用，且向后顺延 100 个端口仍无空闲端口。"
            return 1
        }
        Write-Warn "端口 $actualPort 已被其他程序占用，自动顺延到 $freePort"
        $actualPort = $freePort
    }
    Write-Info "监听地址:    http://${ProbeHost}:$actualPort （bind $BindHost）"

    # 3) 清理旧日志并后台启动 uvicorn
    Remove-Item -LiteralPath $OutLog, $ErrLog -Force -ErrorAction SilentlyContinue
    New-Item -ItemType File -Path $OutLog, $ErrLog -Force | Out-Null

    $env:PIC_POLISH_HOST = $BindHost
    $env:PIC_POLISH_PORT = "$actualPort"

    $arguments = @('-m', 'uvicorn', 'app.main:app', '--host', $BindHost, '--port', "$actualPort")
    Write-Info "启动命令: $Python $($arguments -join ' ')"

    if ($Foreground) {
        Write-Info '前台调试模式（Ctrl+C 结束），不写 .runtime 文件。'
        & $Python @arguments
        return $LASTEXITCODE
    }

    $proc = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $Root `
        -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog `
        -WindowStyle Hidden -PassThru
    $serverPid = $proc.Id
    Write-Info "已后台启动，PID=$serverPid"

    $startedAt = (Get-Date).ToString('o')
    $url = "http://${ProbeHost}:$actualPort"
    Write-Runtime @{
        pid        = $serverPid
        host       = $BindHost
        port       = $actualPort
        url        = $url
        started_at = $startedAt
        engine     = $null
        python     = $Python
        log        = @{ stdout = 'logs/server.log'; stderr = 'logs/server.err.log' }
    }

    # 4) 轮询 /api/health，最多 90s
    Write-Info '等待服务就绪（最多 90 秒）...'
    $health = Wait-Health -BaseUrl $url -TimeoutSeconds 90 -WatchPid $serverPid
    if (-not $health) {
        $alive = [bool](Get-Process -Id $serverPid -ErrorAction SilentlyContinue)
        Write-Fail "服务在 90 秒内未通过健康检查（进程$(if ($alive) { '仍存活' } else { '已退出' })）。"
        if ($alive) {
            Stop-OwnProcessTree -ProcessId $serverPid -Reason '启动超时，清理残留进程' | Out-Null
        }
        Remove-Runtime | Out-Null
        Write-Info '--- logs\server.err.log（最后 25 行）---'
        Show-LogTail -Path $ErrLog -TailLines 25
        Write-Info '--- logs\server.log（最后 15 行）---'
        Show-LogTail -Path $OutLog -TailLines 15
        return 1
    }

    # 5) 回写 engine 等运行信息
    Write-Runtime @{
        pid        = $serverPid
        host       = $BindHost
        port       = $actualPort
        url        = $url
        started_at = $startedAt
        engine     = $health.engine
        python     = $Python
        log        = @{ stdout = 'logs/server.log'; stderr = 'logs/server.err.log' }
    }

    Write-Ok 'PicPolish 启动成功'
    Write-Info "URL:    $url"
    Write-Info "PID:    $serverPid"
    Write-Info "引擎:   $(Format-Engine -Engine $health.engine)"
    Write-Info "日志:   $OutLog"
    Write-Info "        $ErrLog"
    Write-Info "缓存:   $CacheDir"
    Write-Info "停止:   双击 stop.bat 或 pwsh -File scripts\server.ps1 -Action stop"

    if (-not $NoBrowser) {
        try {
            Start-Process $url | Out-Null
            Write-Info '已在默认浏览器中打开页面（使用 -NoBrowser 可关闭）。'
        } catch {
            Write-Warn "自动打开浏览器失败：$($_.Exception.Message)"
        }
    }
    return 0
}

# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------
function Invoke-Stop {
    Ensure-RuntimeDirs
    Write-Title 'PicPolish 停止'
    $runtime = Read-Runtime
    $stopped = 0
    $notRunningReasons = @()

    # 先快照缓存再杀进程：若服务优雅退出（shutdown hook 会清空缓存），
    # 之后再统计就会得到 0，无法报告真实释放字节。
    $cacheBefore = Get-CacheStats

    Update-ProcessTable

    $killedPids = @()
    if ($runtime -and $runtime.pid) {
        $runtimePid = [int]$runtime.pid
        if (Test-OwnProcess -ProcessId $runtimePid) {
            if (Stop-OwnProcessTree -ProcessId $runtimePid -Reason '来自 .runtime\server.json') {
                $killedPids += $runtimePid
                $stopped++
            }
        } else {
            $notRunningReasons += "运行记录中的 PID=$runtimePid 已不再是本项目的 uvicorn 进程（可能已退出或 PID 被复用），不会误杀"
        }
    } else {
        $notRunningReasons += '未找到 .runtime\server.json'
    }

    if ($stopped -gt 0) {
        # 进程终止后刷新快照，再做端口兜底扫描
        Update-ProcessTable
    }

    # 兜底：扫描所有监听端口，凡命令行含本项目路径（或本项目 Python）且含 uvicorn
    # 的进程都终止；监听者可能是解释器转发器的子进程，故沿祖先链向上解析。
    foreach ($listenerPid in Get-ListenerPids) {
        $ownPid = Resolve-OwnProcessPid -ProcessId $listenerPid
        if ($ownPid -le 0) { continue }
        if ($killedPids -contains $ownPid) { continue }
        Write-Info "按端口兜底发现本项目的监听进程 PID=$ownPid（监听 PID=$listenerPid）"
        if (Stop-OwnProcessTree -ProcessId $ownPid -Reason '按监听端口兜底') {
            $killedPids += $ownPid
            $stopped++
        }
    }

    if ($stopped -gt 0) {
        Write-Ok "已停止 PicPolish 服务（共停止 $stopped 个进程：$($killedPids -join ', ')）"
        # 等待端口释放（最多 5 秒）
        if ($runtime -and $runtime.port) {
            $deadline = (Get-Date).AddSeconds(5)
            while ((Get-Date) -lt $deadline) {
                $remaining = Get-NetTCPConnection -State Listen -LocalPort ([int]$runtime.port) -ErrorAction SilentlyContinue
                if (-not $remaining) { break }
                Start-Sleep -Milliseconds 250
            }
        }
    } else {
        Write-Info '服务未在运行。'
        foreach ($reason in $notRunningReasons) { Write-Info "  - $reason" }
    }

    # 清空缓存图片（未运行也执行）
    Write-Title '清理缓存图片'
    $cache = Clear-ImageCache
    $after = Get-CacheStats
    $removedCount = [Math]::Max(0, $cacheBefore.Count - $after.Count)
    $freedBytes = [Math]::Max([long]0, $cacheBefore.Bytes - $after.Bytes)
    Write-Ok "已清空 cache\uploads、cache\results、cache\tmp 下的文件（保留目录）"
    Write-Info "删除文件: $removedCount 个，释放: $(Format-Bytes $freedBytes)"
    if ($cache.Failed -gt 0) {
        Write-Warn "$($cache.Failed) 个文件/目录删除失败（可能正被其他程序占用，稍后可再次运行 stop 清理）"
    }
    Write-Info "当前缓存: $($after.Count) 个文件 / $(Format-Bytes $after.Bytes)"

    # 删除 runtime 文件
    $runtimeRemoved = Remove-Runtime
    if ($runtimeRemoved) { Write-Ok '已删除 .runtime\server.json' } else { Write-Info '无 .runtime\server.json 需要删除' }

    Write-Title '摘要'
    Write-Info "服务状态: 已停止"
    Write-Info "释放空间: $(Format-Bytes $freedBytes)（$removedCount 个文件）"
    Write-Info "运行记录: $(if ($runtimeRemoved) { '已删除' } else { '不存在' })"
    return 0
}

# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
function Invoke-Status {
    Ensure-RuntimeDirs
    Write-Title 'PicPolish 状态'
    $runtime = Read-Runtime
    $running = $false
    $pidValue = 0
    $url = $null

    if ($runtime -and $runtime.pid) {
        $pidValue = [int]$runtime.pid
        if (Test-OwnProcess -ProcessId $pidValue) {
            $running = $true
            $url = if ($runtime.url) { [string]$runtime.url } else { "http://${ProbeHost}:$($runtime.port)" }
        } else {
            Write-Warn "运行记录里的 PID=$pidValue 已不是本项目的 uvicorn 进程（陈旧记录，可运行 stop 清理）"
        }
    }

    $cache = Get-CacheStats
    if ($running) {
        $health = Get-Health -BaseUrl $url
        Write-Ok '服务正在运行'
        Write-Info "URL:    $url"
        Write-Info "PID:    $pidValue"
        if ($runtime.host) { Write-Info "Bind:   $($runtime.host):$($runtime.port)" }
        if ($health) {
            Write-Info "引擎:   $(Format-Engine -Engine $health.engine)"
            Write-Info "版本:   $($health.app) $($health.version)"
        } else {
            Write-Warn '进程存活但 /api/health 无响应（可能正在启动或卡住）'
        }
        if ($runtime.started_at) {
            try {
                $startedValue = $runtime.started_at
                if ($startedValue -is [datetime]) {
                    $started = $startedValue
                } else {
                    $started = [datetime]::Parse([string]$startedValue, [System.Globalization.CultureInfo]::InvariantCulture)
                }
                $span = (Get-Date) - $started
                Write-Info ("启动于:  {0:yyyy-MM-dd HH:mm:ss}（已运行 {1:hh\:mm\:ss}）" -f $started, $span)
            } catch { }
        }
        Write-Info "日志:   $OutLog"
    } else {
        Write-Info '服务未运行。'
    }
    Write-Info "缓存:   $($cache.Count) 个文件 / $(Format-Bytes $cache.Bytes)"
    return 0
}

# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------
function Invoke-Logs {
    Write-Title 'PicPolish 日志'
    Show-LogTail -Path $OutLog -TailLines $Lines
    Show-LogTail -Path $ErrLog -TailLines $Lines
    return 0
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
$exitCode = 0
try {
    switch ($Action) {
        'start' { $exitCode = Invoke-Start }
        'stop' { $exitCode = Invoke-Stop }
        'status' { $exitCode = Invoke-Status }
        'logs' { $exitCode = Invoke-Logs }
        'restart' {
            Invoke-Stop | Out-Null
            $exitCode = Invoke-Start
        }
    }
} catch {
    Write-Fail "未处理的异常：$($_.Exception.Message)"
    Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray
    $exitCode = 1
}
exit $exitCode
