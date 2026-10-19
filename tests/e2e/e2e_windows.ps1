#requires -Version 7.0
<#
.SYNOPSIS
    PicPolish Windows end-to-end acceptance wrapper.

.DESCRIPTION
    Owns the whole server lifecycle so a failed test can never leave a stray
    uvicorn behind, then asserts that the stop script cleaned everything:

      1. start   : pwsh -File scripts\server.ps1 -Action start -NoBrowser -Port <Port>
      2. health  : /api/health reachable, .runtime\server.json content correct
      3. tests   : .venv python tests\e2e\e2e_windows.py   (functional API E2E incl. models API)
      4. quality : .venv python tests\e2e\quality_check.py (real-photo quality/perf)
      5. matrix  : .venv python tests\e2e\model_matrix.py (per-model jobs, denoise matrix, 4K perf)
      6. browser : tests\e2e\browser_smoke.ps1 (headless Edge/Chrome DOM smoke, non-blocking)
      7. dummy   : drop junk files into cache\uploads|results|tmp
      8. stop    : pwsh -File scripts\server.ps1 -Action stop -Port <Port>
      9. verify  : cache empty, port not listening, .runtime\server.json gone,
                   server PID dead, no uvicorn process left for the port

    Results are written to <ArtifactDir>\lifecycle.json; the exit code is 0 only
    when every blocking check passed (browser smoke is recorded as WARN).

.PARAMETER Port
    Test port (default 8796 -- the QA-owned port; devs use 8795/8797/8799).

.PARAMETER ArtifactRoot
    Directory for logs / JSON / quality images. Default: tests\e2e\artifacts\latest
    (wiped at the start of every run).

.EXAMPLE
    pwsh -NoProfile -ExecutionPolicy Bypass -File tests\e2e\e2e_windows.ps1
#>
[CmdletBinding()]
param(
    [int]$Port = 8796,
    [string]$BindHost = '127.0.0.1',
    [string]$Python = '',
    [string]$ArtifactRoot = '',
    [double]$JobTimeout = 300,
    [int]$StartTimeoutSec = 150,
    [switch]$SkipFunctional,
    [switch]$SkipQuality,
    [switch]$SkipMatrix,
    [switch]$Skip4KPerf,
    [switch]$SkipBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $ArtifactRoot) { $ArtifactRoot = Join-Path $PSScriptRoot 'artifacts\latest' }
$ArtifactDir = [System.IO.Path]::GetFullPath($ArtifactRoot)
$psscriptFull = [System.IO.Path]::GetFullPath($PSScriptRoot)
if (-not $ArtifactDir.StartsWith($psscriptFull, [System.StringComparison]::OrdinalIgnoreCase) -and
    -not $ArtifactDir.StartsWith([System.IO.Path]::GetFullPath($ProjectRoot), [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "ArtifactRoot must live inside the project: $ArtifactDir"
}
if (Test-Path $ArtifactDir) { Remove-Item -LiteralPath $ArtifactDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null

$BaseUrl = "http://${BindHost}:$Port"
$ServerPs1 = Join-Path $ProjectRoot 'scripts\server.ps1'
$RuntimeJson = Join-Path $ProjectRoot '.runtime\server.json'
$CacheDir = Join-Path $ProjectRoot 'cache'
$WrapperLog = Join-Path $ArtifactDir 'wrapper.log'
$StartedAt = Get-Date
$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

if (-not $Python) { $Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe' }
$PwshPath = $null
$pwshCmd = Get-Command pwsh -ErrorAction SilentlyContinue
if ($null -ne $pwshCmd) { $PwshPath = $pwshCmd.Source }
if (-not $PwshPath) { $PwshPath = (Get-Process -Id $PID).Path }

$script:CheckResults = [System.Collections.Generic.List[object]]::new()
$script:ServerPid = $null
$script:ServerStarted = $false
$script:ServerStopped = $false
$global:LASTEXITCODE = 0

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Message
    Write-Host $line
    Add-Content -LiteralPath $WrapperLog -Value $line -Encoding utf8
}

function Record-Check {
    param(
        [string]$Name,
        [string]$Status,          # PASS | FAIL | WARN | SKIP
        [string]$Detail = ''
    )
    $script:CheckResults.Add([pscustomobject]@{
        name   = $Name
        status = $Status
        detail = $Detail
    })
    $line = "[$Status] $Name"
    if ($Detail) { $line = "$line  --  $Detail" }
    Write-Log $line
}

function Add-Pass { param([string]$Name, [string]$Detail = '') Record-Check -Name $Name -Status 'PASS' -Detail $Detail }
function Add-Fail { param([string]$Name, [string]$Detail = '') Record-Check -Name $Name -Status 'FAIL' -Detail $Detail }
function Add-Warn { param([string]$Name, [string]$Detail = '') Record-Check -Name $Name -Status 'WARN' -Detail $Detail }
function Add-Skip { param([string]$Name, [string]$Detail = '') Record-Check -Name $Name -Status 'SKIP' -Detail $Detail }

function Test-PortListening {
    param([int]$PortToCheck)
    try {
        $conns = Get-NetTCPConnection -LocalPort $PortToCheck -State Listen -ErrorAction SilentlyContinue
        return [bool]$conns
    } catch {
        # Fallback for environments without the NetTCPIP module.
        $lines = netstat -ano | Select-String -Pattern (":{0}\s+.*LISTENING" -f $PortToCheck)
        return [bool]$lines
    }
}

function Get-CacheFileCount {
    $count = 0
    foreach ($sub in @('uploads', 'results', 'tmp')) {
        $dir = Join-Path $CacheDir $sub
        if (Test-Path -LiteralPath $dir) {
            $files = Get-ChildItem -LiteralPath $dir -Recurse -File -Force -ErrorAction SilentlyContinue
            if ($null -ne $files) { $count += @($files).Count }
        }
    }
    return $count
}

function Get-PortOwnerProcess {
    param([int]$PortToCheck)
    try {
        $conn = Get-NetTCPConnection -LocalPort $PortToCheck -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -eq $conn) { return $null }
        $owner = [int]$conn.OwningProcess
        return Get-CimInstance Win32_Process -Filter "ProcessId = $owner" -ErrorAction SilentlyContinue
    } catch {
        return $null
    }
}

function Get-PortUvicornProcesses {
    param([int]$PortToCheck)
    try {
        return @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
            Where-Object {
                $_.CommandLine -like '*app.main:app*' -and
                $_.CommandLine -like "*--port $PortToCheck*"
            })
    } catch {
        return @()
    }
}

function Get-OtherDevServerPorts {
    param([int]$ExcludePort)
    $ports = @()
    foreach ($otherPort in @(8765, 8795, 8797, 8798, 8799)) {
        if ($otherPort -eq $ExcludePort) { continue }
        if (Test-PortListening -PortToCheck $otherPort) { $ports += $otherPort }
    }
    return $ports
}

function Get-ForeignInstance {
    # server.ps1's start is idempotent: if .runtime/server.json points at a live
    # instance, `start -Port 8796` would just print that instance's URL.  Testing
    # somebody else's server (and later stopping it) is not acceptable, so detect
    # it up-front.  A leftover instance on *our* port is handled by the normal
    # pre-start stop path.
    if (-not (Test-Path -LiteralPath $RuntimeJson)) { return $null }
    $runtime = Read-JsonSafe -Path $RuntimeJson
    if ($null -eq $runtime) { return $null }
    $foreignPort = 0
    try { $foreignPort = [int]$runtime.port } catch { return $null }
    if ($foreignPort -eq $Port) { return $null }
    $foreignPid = 0
    try { $foreignPid = [int]$runtime.pid } catch { }
    $alive = $false
    if ($foreignPid -gt 0 -and $null -ne (Get-Process -Id $foreignPid -ErrorAction SilentlyContinue)) {
        $alive = $true
    }
    if (-not $alive -and $foreignPort -gt 0 -and (Test-PortListening -PortToCheck $foreignPort)) {
        $alive = $true
    }
    if (-not $alive) { return $null }
    return "another PicPolish instance is alive on port $foreignPort (pid=$foreignPid, $RuntimeJson); " +
        "stop it (or wait for its owner) before running acceptance"
}

function Invoke-PwshFile {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$StdOut,
        [string]$StdErr,
        [int]$TimeoutSec
    )
    $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$FilePath`"") + $Arguments
    $proc = Start-Process -FilePath $PwshPath -ArgumentList $argList -PassThru -NoNewWindow `
        -RedirectStandardOutput $StdOut -RedirectStandardError $StdErr
    if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
        try { $proc.Kill($true) } catch { }
        return -1
    }
    return $proc.ExitCode
}

function Get-PythonLogTail {
    param([string]$Path, [int]$Lines = 40)
    if (-not (Test-Path -LiteralPath $Path)) { return "(no log: $Path)" }
    return (Get-Content -LiteralPath $Path -Tail $Lines -Encoding utf8) -join "`n"
}

function Invoke-PythonStep {
    param(
        [string]$StepName,
        [string]$RelativeScript,
        [string[]]$ScriptArguments,
        [int]$TimeoutSec
    )
    $scriptPath = Join-Path $ProjectRoot $RelativeScript
    $logPath = Join-Path $ArtifactDir ("{0}.log" -f $StepName)
    $errPath = Join-Path $ArtifactDir ("{0}.err.log" -f $StepName)
    Write-Log "--- running $StepName : python $RelativeScript ---"
    $argList = @("`"$scriptPath`"") + @($ScriptArguments | ForEach-Object { "`"$_`"" })
    $proc = Start-Process -FilePath $Python -ArgumentList $argList -PassThru -NoNewWindow `
        -RedirectStandardOutput $logPath -RedirectStandardError $errPath
    $finished = $proc.WaitForExit($TimeoutSec * 1000)
    if (-not $finished) {
        try { $proc.Kill($true) } catch { }
        Write-Log "$StepName TIMED OUT after ${TimeoutSec}s (killed)"
        Write-Log (Get-PythonLogTail -Path $logPath -Lines 30)
        return -1
    }
    Write-Log "$StepName exit code = $($proc.ExitCode)"
    Write-Log (Get-PythonLogTail -Path $logPath -Lines 8)
    if ($proc.ExitCode -ne 0) {
        Write-Log "--- $StepName log tail (failure) ---"
        Write-Log (Get-PythonLogTail -Path $logPath -Lines 60)
        if (Test-Path -LiteralPath $errPath) {
            Write-Log "--- $StepName stderr tail ---"
            Write-Log (Get-PythonLogTail -Path $errPath -Lines 40)
        }
    }
    return $proc.ExitCode
}

function Read-JsonSafe {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try {
        $raw = Get-Content -LiteralPath $Path -Raw -Encoding utf8
        if (-not $raw) { return $null }
        return $raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Get-EngineSnapshot {
    try {
        $health = Invoke-RestMethod -Uri "$BaseUrl/api/health" -TimeoutSec 5
        return [pscustomobject]@{
            status = $health.status
            app    = $health.app
            version = $health.version
            engine = $health.engine
            limits = $health.limits
        }
    } catch {
        return $null
    }
}

# =========================================================================
# 0. Preconditions
# =========================================================================
Write-Log "=============================================================="
Write-Log "PicPolish Windows E2E lifecycle wrapper"
Write-Log "project  : $ProjectRoot"
Write-Log "base url : $BaseUrl"
Write-Log "artifacts: $ArtifactDir"
Write-Log "=============================================================="

if (-not (Test-Path -LiteralPath $Python)) {
    Add-Fail 'precondition.python' "venv python not found: $Python (run scripts\setup_env.ps1)"
}
if (-not (Test-Path -LiteralPath $ServerPs1)) {
    Add-Fail 'precondition.server_ps1' "missing $ServerPs1"
}
foreach ($sample in @('photo_1280x720.jpg', 'text_gradient_1280x720.png',
        'portrait_720x1280.jpg', 'alpha_800x600.png', 'tiny_320x240.png',
        'anime_1280x720.png', 'noisy_gauss_1280x720.jpg')) {
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "samples\$sample"))) {
        Add-Fail 'precondition.samples' "missing samples\$sample (run tests\e2e\make_samples.py)"
    }
}
# A live instance on another port would make `start` return it idempotently.
$foreignInstance = Get-ForeignInstance
if ($null -ne $foreignInstance) {
    Add-Fail 'precondition.no_foreign_instance' $foreignInstance
}
# Exclusivity: other dev servers share cache/ and any concurrent stop script
# wipes cache/uploads mid-job (observed: "无法读取上传图片").  Refuse to start.
$busyDevPorts = @(Get-OtherDevServerPorts -ExcludePort $Port)
if ($busyDevPorts.Count -gt 0) {
    Add-Fail 'precondition.exclusive_env' `
        ("other PicPolish dev server(s) listening on port(s) {0}; stop them before acceptance (shared cache + single-instance lock)" -f ($busyDevPorts -join ','))
} else {
    Add-Pass 'precondition.exclusive_env' 'no other PicPolish dev servers listening'
}
$preconditionFailed = @($script:CheckResults | Where-Object { $_.status -eq 'FAIL' }).Count -gt 0
$functionalJson = Join-Path $ArtifactDir 'functional\e2e_windows.json'
$qualityJson = Join-Path $ArtifactDir 'quality\quality_check.json'
$matrixJson = Join-Path $ArtifactDir 'matrix\model_matrix.json'
$lifecycleJson = Join-Path $ArtifactDir 'lifecycle.json'
$browserJson = Join-Path $ArtifactDir 'browser\browser_smoke.json'

$exitCode = 0
try {
    if ($preconditionFailed) {
        Add-Skip 'lifecycle' 'aborted: preconditions failed'
    }
    else {
        # =================================================================
        # 1. Start the server through the real start script
        # =================================================================
        if (Test-PortListening -Port $Port) {
            Write-Log "port $Port is already busy before start -- trying stop first"
            $stopOut = Join-Path $ArtifactDir 'prestart-stop.log'
            $stopErr = Join-Path $ArtifactDir 'prestart-stop.err.log'
            $preCode = Invoke-PwshFile -FilePath $ServerPs1 -Arguments @('-Action', 'stop', '-Port', "$Port") `
                -StdOut $stopOut -StdErr $stopErr -TimeoutSec 90
            Write-Log "pre-start stop exit code = $preCode"
        }
        if (Test-PortListening -Port $Port) {
            Add-Fail 'lifecycle.port_free' "port $Port is still listening before start; refusing to test against a foreign server"
        }
        else {
            Add-Pass 'lifecycle.port_free' "port $Port is free"
            $startOut = Join-Path $ArtifactDir 'start-server.log'
            $startErr = Join-Path $ArtifactDir 'start-server.err.log'
            $startCode = Invoke-PwshFile -FilePath $ServerPs1 `
                -Arguments @('-Action', 'start', '-Port', "$Port", '-Host', $BindHost, '-NoBrowser') `
                -StdOut $startOut -StdErr $startErr -TimeoutSec $StartTimeoutSec
            if ($startCode -eq 0) {
                Add-Pass 'lifecycle.start_exit_code' "scripts\server.ps1 start exited 0"
            } else {
                Add-Fail 'lifecycle.start_exit_code' "start exit code $startCode (see $startOut)"
            }
            Write-Log (Get-PythonLogTail -Path $startOut -Lines 25)
            if (Test-Path -LiteralPath $startErr) {
                $errText = Get-Content -LiteralPath $startErr -Raw -Encoding utf8
                if ($errText -and $errText.Trim()) { Write-Log "start stderr: $($errText.Trim())" }
            }
        }

        # =================================================================
        # 2. Runtime file + health
        # =================================================================
        if (Test-Path -LiteralPath $RuntimeJson) {
            Add-Pass 'lifecycle.runtime_json_created' $RuntimeJson
            $runtime = Read-JsonSafe -Path $RuntimeJson
            if ($null -eq $runtime) {
                Add-Fail 'lifecycle.runtime_json_parse' 'file exists but is not valid JSON'
            } else {
                $script:ServerPid = [int]$runtime.pid
                $script:ServerStarted = $true
                $runtimePort = [int]$runtime.port
                if ($runtimePort -eq $Port) {
                    Add-Pass 'lifecycle.runtime_json_port' "port=$runtimePort"
                } else {
                    Add-Warn 'lifecycle.runtime_json_port' "start script shifted port: asked $Port, got $runtimePort (using actual)"
                    $Port = $runtimePort
                    $BaseUrl = "http://${BindHost}:$Port"
                }
                if ($runtime.url -eq $BaseUrl -or "$($runtime.url)".EndsWith(":$Port")) {
                    Add-Pass 'lifecycle.runtime_json_url' "url=$($runtime.url)"
                } else {
                    Add-Warn 'lifecycle.runtime_json_url' "url=$($runtime.url) (expected $BaseUrl)"
                }
                if ($script:ServerPid -gt 0) {
                    Add-Pass 'lifecycle.runtime_json_pid' "pid=$($script:ServerPid)"
                } else {
                    Add-Fail 'lifecycle.runtime_json_pid' "invalid pid: $($runtime.pid)"
                }
            }
        } else {
            Add-Fail 'lifecycle.runtime_json_created' "missing $RuntimeJson after start"
        }

        $healthOk = $false
        for ($attempt = 1; $attempt -le 30; $attempt++) {
            try {
                $health = Invoke-RestMethod -Uri "$BaseUrl/api/health" -TimeoutSec 5
                if ($health.status -eq 'ok') { $healthOk = $true; break }
            } catch {
                Start-Sleep -Milliseconds 500
            }
        }
        if ($healthOk) {
            $engine = Invoke-RestMethod -Uri "$BaseUrl/api/health" -TimeoutSec 5
            Add-Pass 'lifecycle.health' ("backend={0} device={1} gpu={2}" -f $engine.engine.backend, $engine.engine.device, $engine.engine.gpu)
        } else {
            Add-Fail 'lifecycle.health' "/api/health not OK at $BaseUrl"
            Add-Skip 'functional' '/api/health unreachable'
            Add-Skip 'quality' '/api/health unreachable'
            Add-Skip 'browser' '/api/health unreachable'
        }

        if ($healthOk) {
            # A freshly started server must have empty cache dirs.  The cache is
            # shared by every PicPolish instance, so if a dev server is still
            # running on one of the other reserved ports we only WARN.
            $freshCount = Get-CacheFileCount
            if ($freshCount -eq 0) {
                Add-Pass 'lifecycle.cache_empty_after_start' '0 files in cache\uploads|results|tmp'
            } else {
                $otherServers = @(Get-OtherDevServerPorts -ExcludePort $Port)
                if ($otherServers.Count -gt 0) {
                    Add-Warn 'lifecycle.cache_empty_after_start' `
                        ("{0} files in cache, but other PicPolish dev server(s) are running on port(s) {1} (shared cache)" -f $freshCount, ($otherServers -join ','))
                } else {
                    Add-Fail 'lifecycle.cache_empty_after_start' "$freshCount files present after a fresh start"
                }
            }
            # Interference canary: another process clearing the shared cache
            # (e.g. a teammate's concurrent stop script) deletes this marker.
            $markerPath = Join-Path $CacheDir 'tmp\qa_e2e_marker.bin'
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markerPath) | Out-Null
            [System.IO.File]::WriteAllBytes($markerPath, [byte[]](1..16))
            Add-Pass 'lifecycle.cache_marker_created' 'cache/tmp/qa_e2e_marker.bin (concurrency canary)'

            # =============================================================
            # 3. Functional E2E
            # =============================================================
            if ($SkipFunctional) {
                Add-Skip 'functional' 'skipped by -SkipFunctional'
            } else {
                $funcExit = Invoke-PythonStep -StepName 'functional' -RelativeScript 'tests\e2e\e2e_windows.py' `
                    -ScriptArguments @('--base-url', $BaseUrl, '--artifact-dir', (Join-Path $ArtifactDir 'functional'),
                        '--json-out', $functionalJson, '--job-timeout', "$JobTimeout",
                        '--expect-fresh', '--expect-engine', 'realesrgan', '--expect-device', 'cuda') `
                    -TimeoutSec ([int]($JobTimeout + 180))
                if ($funcExit -eq 0) { Add-Pass 'functional.exit_code' 'e2e_windows.py exited 0' }
                else { Add-Fail 'functional.exit_code' "e2e_windows.py exited $funcExit" }
                $funcSummary = Read-JsonSafe -Path $functionalJson
                if ($null -ne $funcSummary -and $null -ne $funcSummary.summary) {
                    Add-Pass 'functional.json' ("PASS={0} FAIL={1} WARN={2} SKIP={3}" -f $funcSummary.summary.pass, $funcSummary.summary.fail, $funcSummary.summary.warn, $funcSummary.summary.skip)
                    if ([int]$funcSummary.summary.fail -gt 0) {
                        Add-Fail 'functional.checks' ("failed: " + ($funcSummary.summary.failed_ids -join ', '))
                    }
                } else {
                    Add-Fail 'functional.json' "missing/invalid $functionalJson"
                }
                # Detect a teammate starting/stopping their own instance mid-run:
                # a concurrent stop script wipes the shared cache and breaks uploads.
                $concurrentNow = @(Get-OtherDevServerPorts -ExcludePort $Port)
                if ($concurrentNow.Count -gt 0) {
                    Add-Warn 'lifecycle.concurrent_dev_servers' `
                        ("dev server(s) listening on port(s) {0} during the run (shared cache risk)" -f ($concurrentNow -join ','))
                }
                $markerPath = Join-Path $CacheDir 'tmp\qa_e2e_marker.bin'
                if (-not (Test-Path -LiteralPath $markerPath)) {
                    Add-Warn 'lifecycle.cache_interference' `
                        'cache/tmp marker disappeared mid-run -- another process cleared the shared cache (concurrent stop script?)'
                }
            }

            # =============================================================
            # 4. Quality + performance
            # =============================================================
            if ($SkipQuality) {
                Add-Skip 'quality' 'skipped by -SkipQuality'
            } else {
                $qualityExit = Invoke-PythonStep -StepName 'quality' -RelativeScript 'tests\e2e\quality_check.py' `
                    -ScriptArguments @('--base-url', $BaseUrl, '--artifact-dir', (Join-Path $ArtifactDir 'quality'),
                        '--json-out', $qualityJson, '--job-timeout', '180') `
                    -TimeoutSec 600
                if ($qualityExit -eq 0) { Add-Pass 'quality.exit_code' 'quality_check.py exited 0' }
                else { Add-Fail 'quality.exit_code' "quality_check.py exited $qualityExit" }
                $qualitySummary = Read-JsonSafe -Path $qualityJson
                if ($null -ne $qualitySummary -and $null -ne $qualitySummary.summary) {
                    Add-Pass 'quality.json' ("PASS={0} FAIL={1} WARN={2}" -f $qualitySummary.summary.pass, $qualitySummary.summary.fail, $qualitySummary.summary.warn)
                    if ([int]$qualitySummary.summary.fail -gt 0) {
                        Add-Fail 'quality.checks' ("failed: " + ($qualitySummary.summary.failed_ids -join ', '))
                    }
                    foreach ($case in @($qualitySummary.cases)) {
                        if ($null -eq $case) { continue }
                        if ($null -ne $case.PSObject.Properties['sharpness']) {
                            $sharpness = $case.sharpness
                            Add-Pass ("quality.{0}.sharpness" -f $case.label) ("Laplacian SR={0} vs Lanczos={1} ({2}x)" -f $sharpness.laplacian_sr, $sharpness.laplacian_lanczos, $sharpness.ratio_vs_lanczos)
                        }
                        if ($null -ne $case.PSObject.Properties['perf']) {
                            $perf = $case.perf
                            Add-Pass 'quality.perf.720p_to_1080p' ("elapsed_sec={0}s wall={1}s backend={2} target<=45s" -f $perf.elapsed_sec, $perf.wall_sec, $perf.backend)
                        }
                    }
                } else {
                    Add-Fail 'quality.json' "missing/invalid $qualityJson"
                }
            }

            # =============================================================
            # 5. Model matrix (multi-model jobs + denoise matrix + 4K perf)
            # =============================================================
            if ($SkipMatrix) {
                Add-Skip 'matrix' 'skipped by -SkipMatrix'
            } else {
                $matrixArgs = @('--base-url', $BaseUrl,
                    '--artifact-dir', (Join-Path $ArtifactDir 'matrix'),
                    '--json-out', $matrixJson, '--job-timeout', "$JobTimeout")
                if ($Skip4KPerf) { $matrixArgs += '--no-4k' }
                $matrixTimeout = if ($Skip4KPerf) { 1200 } else { 2400 }
                $matrixExit = Invoke-PythonStep -StepName 'matrix' -RelativeScript 'tests\e2e\model_matrix.py' `
                    -ScriptArguments $matrixArgs -TimeoutSec $matrixTimeout
                if ($matrixExit -eq 0) { Add-Pass 'matrix.exit_code' 'model_matrix.py exited 0' }
                else { Add-Fail 'matrix.exit_code' "model_matrix.py exited $matrixExit" }
                $matrixSummary = Read-JsonSafe -Path $matrixJson
                if ($null -ne $matrixSummary -and $null -ne $matrixSummary.summary) {
                    Add-Pass 'matrix.json' ("PASS={0} FAIL={1} WARN={2} SKIP={3}" -f $matrixSummary.summary.pass, $matrixSummary.summary.fail, $matrixSummary.summary.warn, $matrixSummary.summary.skip)
                    if ([int]$matrixSummary.summary.fail -gt 0) {
                        Add-Fail 'matrix.checks' ("failed: " + ($matrixSummary.summary.failed_ids -join ', '))
                    }
                    if ($null -ne $matrixSummary.PSObject.Properties['perf']) {
                        foreach ($perfProp in $matrixSummary.perf.PSObject.Properties) {
                            $perfRow = $perfProp.Value
                            if ($null -eq $perfRow) { continue }
                            $p720 = $perfRow.PSObject.Properties['720p_sec'].Value
                            $p720w = $perfRow.PSObject.Properties['720p_wall_sec'].Value
                            $p4k = $perfRow.PSObject.Properties['4k_sec'].Value
                            $p4kw = $perfRow.PSObject.Properties['4k_wall_sec'].Value
                            Add-Pass ("matrix.perf.{0}" -f $perfProp.Name) ("720p->1080p={0}s (wall {1}s)  4K={2}s (wall {3}s)" -f $p720, $p720w, $p4k, $p4kw)
                        }
                    }
                    if ($null -ne $matrixSummary.PSObject.Properties['diffs']) {
                        $diffs = $matrixSummary.diffs
                        Add-Pass 'matrix.diffs' ("anime6b_vs_x4plus={0}  rrdb_denoise_on_off={1}  cugan_L0_vs_L3={2}" -f $diffs.anime6b_vs_x4plus_mean_abs, $diffs.rrdb_denoise_on_off_mean_abs, $diffs.cugan_level0_vs_level3_mean_abs)
                    }
                } else {
                    Add-Fail 'matrix.json' "missing/invalid $matrixJson"
                }
            }

            # =============================================================
            # 6. Browser smoke (non-blocking)
            # =============================================================
            if ($SkipBrowser) {
                Add-Skip 'browser' 'skipped by -SkipBrowser'
            } else {
                $browserPs1 = Join-Path $PSScriptRoot 'browser_smoke.ps1'
                if (-not (Test-Path -LiteralPath $browserPs1)) {
                    Add-Warn 'browser.smoke' "missing $browserPs1"
                } else {
                    $browserOut = Join-Path $ArtifactDir 'browser\stdout.log'
                    $browserErr = Join-Path $ArtifactDir 'browser\stderr.log'
                    New-Item -ItemType Directory -Force -Path (Join-Path $ArtifactDir 'browser') | Out-Null
                    $browserCode = Invoke-PwshFile -FilePath $browserPs1 `
                        -Arguments @('-Url', "$BaseUrl/", '-ArtifactDir', (Join-Path $ArtifactDir 'browser')) `
                        -StdOut $browserOut -StdErr $browserErr -TimeoutSec 180
                    Write-Log (Get-PythonLogTail -Path $browserOut -Lines 20)
                    $browserSummary = Read-JsonSafe -Path $browserJson
                    if ($browserCode -eq 0) {
                        Add-Pass 'browser.smoke' 'headless DOM checks passed'
                    } else {
                        Add-Warn 'browser.smoke' "exit code $browserCode (non-blocking; see $browserOut)"
                    }
                    if ($null -ne $browserSummary -and $null -ne $browserSummary.summary) {
                        Add-Pass 'browser.json' ("PASS={0} FAIL={1}" -f $browserSummary.summary.pass, $browserSummary.summary.fail)
                    } elseif ($browserCode -eq 2) {
                        Add-Warn 'browser.json' 'no Edge/Chrome found on this machine'
                    }
                }
            }
        }

        # =================================================================
        # 7. Cache cleanup fixture, then stop through the real stop script
        # =================================================================
        if ($script:ServerStarted) {
            foreach ($sub in @('uploads', 'results', 'tmp')) {
                $dir = Join-Path $CacheDir $sub
                New-Item -ItemType Directory -Force -Path $dir | Out-Null
                [System.IO.File]::WriteAllBytes((Join-Path $dir 'qa_dummy.bin'), [byte[]](1..64))
            }
            Add-Pass 'lifecycle.cache_dummy_created' '3 dummy files created before stop'

            $stopOut = Join-Path $ArtifactDir 'stop-server.log'
            $stopErr = Join-Path $ArtifactDir 'stop-server.err.log'
            $stopMeasure = [System.Diagnostics.Stopwatch]::StartNew()
            $stopCode = Invoke-PwshFile -FilePath $ServerPs1 -Arguments @('-Action', 'stop', '-Port', "$Port") `
                -StdOut $stopOut -StdErr $stopErr -TimeoutSec 120
            $stopMeasure.Stop()
            $stopSec = [Math]::Round($stopMeasure.Elapsed.TotalSeconds, 2)
            if ($stopCode -eq 0) { Add-Pass 'lifecycle.stop_exit_code' 'scripts\server.ps1 stop exited 0' }
            else { Add-Fail 'lifecycle.stop_exit_code' "stop exit code $stopCode" }
            if ($stopSec -gt 20.0) {
                Add-Warn 'lifecycle.stop_duration' "${stopSec}s (slow stop -- port-fallback path?)"
            } else {
                Add-Pass 'lifecycle.stop_duration' "${stopSec}s"
            }
            $script:ServerStopped = $true
            Write-Log (Get-PythonLogTail -Path $stopOut -Lines 25)

            # ---- post-stop assertions -----------------------------------
            $settled = $false
            for ($attempt = 1; $attempt -le 20; $attempt++) {
                if (-not (Test-PortListening -Port $Port)) { $settled = $true; break }
                Start-Sleep -Milliseconds 500
            }
            if ($settled) { Add-Pass 'lifecycle.port_closed' "port $Port not listening" }
            else { Add-Fail 'lifecycle.port_closed' "port $Port is still listening after stop" }

            $remaining = Get-CacheFileCount
            if ($remaining -eq 0) {
                Add-Pass 'lifecycle.cache_cleared' 'cache\uploads|results|tmp are empty after stop'
            } else {
                $otherServers = @(Get-OtherDevServerPorts -ExcludePort $Port)
                if ($otherServers.Count -gt 0) {
                    Add-Warn 'lifecycle.cache_cleared' `
                        ("{0} files remain, but other PicPolish dev server(s) are running on port(s) {1} (shared cache)" -f $remaining, ($otherServers -join ','))
                } else {
                    Add-Fail 'lifecycle.cache_cleared' "$remaining files remain in cache after stop"
                }
            }

            if (Test-Path -LiteralPath $RuntimeJson) {
                Add-Fail 'lifecycle.runtime_json_removed' "still exists: $RuntimeJson"
            } else {
                Add-Pass 'lifecycle.runtime_json_removed' 'runtime file deleted'
            }

            if ($null -ne $script:ServerPid) {
                $alive = Get-Process -Id $script:ServerPid -ErrorAction SilentlyContinue
                if ($null -eq $alive) { Add-Pass 'lifecycle.server_pid_dead' "pid $($script:ServerPid) gone" }
                else { Add-Fail 'lifecycle.server_pid_dead' "pid $($script:ServerPid) still running" }
            }

            $leftover = @(Get-PortUvicornProcesses -PortToCheck $Port)
            if ($leftover.Count -eq 0) {
                Add-Pass 'lifecycle.no_uvicorn_residue' "no python app.main:app process for port $Port"
            } else {
                Add-Fail 'lifecycle.no_uvicorn_residue' ("PIDs: " + (($leftover | ForEach-Object { $_.ProcessId }) -join ', '))
            }
        } else {
            Add-Skip 'lifecycle.stop' 'server never started'
        }
    }
} catch {
    Add-Fail 'wrapper.exception' ("{0}: {1}" -f $_.Exception.GetType().Name, $_.Exception.Message)
    Write-Log "STACK: $($_.ScriptStackTrace)"
} finally {
    # Safety net: never leave our own server running, whatever happened above.
    if ($script:ServerStarted -and -not $script:ServerStopped) {
        # Only stop when the runtime file / listening port actually belongs to
        # this run's port; never touch another agent's instance on a dev port.
        $ownRuntime = $false
        if (Test-Path -LiteralPath $RuntimeJson) {
            $emRuntime = Read-JsonSafe -Path $RuntimeJson
            if ($null -ne $emRuntime) {
                try { $ownRuntime = [int]$emRuntime.port -eq $Port } catch { $ownRuntime = $false }
            }
        }
        $ownProcesses = @(Get-PortUvicornProcesses -PortToCheck $Port)
        if (-not $ownRuntime -and $ownProcesses.Count -eq 0) {
            Write-Log 'emergency cleanup: nothing owned by this run is alive; not stopping other instances'
        }
        else {
            Write-Log 'emergency cleanup: server still running, invoking stop'
            $emOut = Join-Path $ArtifactDir 'emergency-stop.log'
            $emErr = Join-Path $ArtifactDir 'emergency-stop.err.log'
            $emCode = -1
            try {
                $emCode = Invoke-PwshFile -FilePath $ServerPs1 -Arguments @('-Action', 'stop', '-Port', "$Port") `
                    -StdOut $emOut -StdErr $emErr -TimeoutSec 120
            } catch {
                Write-Log "emergency stop failed: $($_.Exception.Message)"
            }
            Write-Log "emergency stop exit code = $emCode"
            if (Test-PortListening -Port $Port) {
                $owner = Get-PortOwnerProcess -PortToCheck $Port
                if ($null -ne $owner -and "$($owner.CommandLine)" -like '*app.main:app*') {
                    Write-Log "killing verified uvicorn PID $($owner.ProcessId) on port $Port"
                    & taskkill.exe /PID $owner.ProcessId /T /F 2>&1 | Out-Null
                    $global:LASTEXITCODE = 0
                } else {
                    Write-Log "port $Port still busy but owner is not this project's uvicorn -- not killing"
                }
            }
        }
    }
    $Stopwatch.Stop()
    $duration = [Math]::Round($Stopwatch.Elapsed.TotalSeconds, 2)
    $pass = @($script:CheckResults | Where-Object { $_.status -eq 'PASS' }).Count
    $failList = @($script:CheckResults | Where-Object { $_.status -eq 'FAIL' })
    $warn = @($script:CheckResults | Where-Object { $_.status -eq 'WARN' }).Count
    $skip = @($script:CheckResults | Where-Object { $_.status -eq 'SKIP' }).Count
    $summary = [pscustomobject]@{
        pass = $pass; fail = $failList.Count; warn = $warn; skip = $skip
        failed_names = @($failList | ForEach-Object { $_.name })
    }
    $engineSnapshot = Get-EngineSnapshot
    $functionalSummary = Read-JsonSafe -Path $functionalJson
    $qualitySummary = Read-JsonSafe -Path $qualityJson
    $matrixSummary = Read-JsonSafe -Path $matrixJson
    $browserSummary = Read-JsonSafe -Path $browserJson
    $payload = [pscustomobject]@{
        suite = 'windows-lifecycle'
        port = $Port
        base_url = $BaseUrl
        project_root = $ProjectRoot
        started_at = $StartedAt.ToString('o')
        duration_sec = $duration
        engine = $engineSnapshot
        functional = $functionalSummary
        quality = $qualitySummary
        matrix = $matrixSummary
        browser = $browserSummary
        checks = @($script:CheckResults)
        summary = $summary
        artifacts = $ArtifactDir
    }
    try {
        $json = $payload | ConvertTo-Json -Depth 12
        Set-Content -LiteralPath $lifecycleJson -Value $json -Encoding utf8
        Write-Log "lifecycle JSON: $lifecycleJson"
    } catch {
        Write-Log "failed to write lifecycle JSON: $($_.Exception.Message)"
    }
    Write-Log '=============================================================='
    Write-Log ("WINDOWS E2E SUMMARY: PASS={0} FAIL={1} WARN={2} SKIP={3} elapsed={4}s" -f $pass, $failList.Count, $warn, $skip, $duration)
    if ($failList.Count -gt 0) {
        Write-Log ("FAILED: " + (($failList | ForEach-Object { $_.name }) -join ', '))
    }
    Write-Log "artifacts: $ArtifactDir"
    Write-Log '=============================================================='
    $exitCode = if ($failList.Count -eq 0) { 0 } else { 1 }
}

exit $exitCode
