#requires -Version 7.0
<#
.SYNOPSIS
    Headless browser DOM smoke test for the PicPolish web UI.

.DESCRIPTION
    Finds every installed Chromium-based browser (Edge and Chrome), loads the
    running PicPolish page with
    ``--headless=new --dump-dom --virtual-time-budget=8000`` and asserts the key
    DOM elements: document title, upload input/drop zone, the four tier cards,
    result-save controls ("另存为"/download), engine badge, the app.js
    reference, and (v1.1) the model selection cards, recommendation text and
    denoise controls.  A screenshot per browser is captured as report evidence.

    This is a best-effort check: the Windows wrapper records failures as WARN,
    they do not block acceptance.  All installed browsers are attempted (not
    just the first one).

    Exit codes: 0 = at least one browser passed every required check,
    1 = required check failed, 2 = no supported browser found,
    3 = browser(s) produced no DOM.

.EXAMPLE
    pwsh -NoProfile -ExecutionPolicy Bypass -File tests\e2e\browser_smoke.ps1 `
        -Url http://127.0.0.1:8796/ -ArtifactDir tests\e2e\artifacts\latest\browser
#>
[CmdletBinding()]
param(
    [string]$Url = 'http://127.0.0.1:8796/',
    [string]$ArtifactDir = '',
    [int]$VirtualTimeBudgetMs = 8000,
    [int]$TimeoutSec = 90
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $ArtifactDir) {
    $ArtifactDir = Join-Path ([System.IO.Path]::GetTempPath()) ("picpolish_browser_" + [guid]::NewGuid().ToString('N').Substring(0, 8))
}
New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null
$ArtifactDir = [System.IO.Path]::GetFullPath($ArtifactDir)

$script:Checks = [System.Collections.Generic.List[object]]::new()

function Add-Check {
    param([string]$Name, [bool]$Ok, [string]$Detail = '', [bool]$Required = $true)
    $status = if ($Ok) { 'PASS' } elseif ($Required) { 'FAIL' } else { 'WARN' }
    $script:Checks.Add([pscustomobject]@{ name = $Name; status = $status; detail = $Detail; required = $Required })
    $line = "[$status] $Name"
    if ($Detail) { $line = "$line  --  $Detail" }
    Write-Host $line
}

# ---------------------------------------------------------------------------
# Locate Chromium-based browsers (Edge first: always present on Win10/11)
# ---------------------------------------------------------------------------
$candidates = @(
    @{ Name = 'Edge';   Path = (Join-Path ${env:ProgramFiles(x86)} 'Microsoft\Edge\Application\msedge.exe') },
    @{ Name = 'Edge';   Path = (Join-Path $env:ProgramFiles 'Microsoft\Edge\Application\msedge.exe') },
    @{ Name = 'Chrome'; Path = (Join-Path $env:ProgramFiles 'Google\Chrome\Application\chrome.exe') },
    @{ Name = 'Chrome'; Path = (Join-Path ${env:ProgramFiles(x86)} 'Google\Chrome\Application\chrome.exe') },
    @{ Name = 'Chrome'; Path = (Join-Path $env:LOCALAPPDATA 'Google\Chrome\Application\chrome.exe') }
)
$browsers = @()
foreach ($candidate in $candidates) {
    if ($candidate.Path -and (Test-Path -LiteralPath $candidate.Path)) {
        if (-not ($browsers | Where-Object { $_.Path -eq $candidate.Path })) {
            $browsers += [pscustomobject]@{ Name = $candidate.Name; Path = $candidate.Path }
        }
    }
}

function Remove-UserDataDir {
    param([string]$Path)
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop }
            return
        } catch {
            Start-Sleep -Milliseconds 400
        }
    }
}

function Invoke-Headless {
    param(
        [string]$Exe,
        [string[]]$BrowserArguments,
        [string]$StdOutPath,
        [string]$StdErrPath
    )
    $userData = Join-Path $ArtifactDir ("ud-" + [guid]::NewGuid().ToString('N').Substring(0, 10))
    $argList = @(
        '--headless=new',
        '--disable-gpu',
        '--no-first-run',
        '--no-default-browser-check',
        '--disable-extensions',
        '--disable-dev-shm-usage',
        '--hide-scrollbars',
        '--mute-audio',
        '--force-device-scale-factor=1',
        "--virtual-time-budget=$VirtualTimeBudgetMs",
        "--user-data-dir=`"$userData`""
    ) + $BrowserArguments + @("`"$Url`"")
    $proc = Start-Process -FilePath $Exe -ArgumentList $argList -PassThru -NoNewWindow `
        -RedirectStandardOutput $StdOutPath -RedirectStandardError $StdErrPath
    $finished = $proc.WaitForExit($TimeoutSec * 1000)
    if (-not $finished) {
        try { $proc.Kill($true) } catch { }
        $code = -1
    } else {
        $code = $proc.ExitCode
    }
    Remove-UserDataDir -Path $userData
    return $code
}

function Get-DomTitle {
    param([string]$Dom)
    $titleMatch = [regex]::Match($Dom, '(?is)<title[^>]*>(.*?)</title>')
    if ($titleMatch.Success) { return $titleMatch.Groups[1].Value.Trim() }
    return ''
}

function Test-Dom {
    param([string]$BrowserName, [string]$Dom)
    $prefix = "browser.$BrowserName"
    $lower = $Dom.ToLowerInvariant()

    Add-Check -Name "$prefix.dom_captured" -Ok ($Dom.Length -gt 500) -Detail ("{0} chars" -f $Dom.Length)

    $titleText = Get-DomTitle -Dom $Dom
    Add-Check -Name "$prefix.title" -Ok ($titleText -match '(?i)picpolish') -Detail ("<title>=$titleText")
    Add-Check -Name "$prefix.title_text" -Ok ($titleText.Length -gt 0) -Detail $titleText -Required $false

    Add-Check -Name "$prefix.upload_area" -Ok `
        ($lower.Contains('type="file"') -or $lower.Contains("type='file'") -or
        $Dom.Contains('上传') -or $Dom.Contains('拖拽') -or $lower.Contains('dropzone') -or $lower.Contains('drop-zone')) `
        -Detail 'file input / upload / drop zone present'

    $missingTiers = @()
    foreach ($tier in @('720P', '1080P', '2K', '4K')) {
        if (-not ($Dom -match ('(?i)' + [regex]::Escape($tier) + '\b'))) { $missingTiers += $tier }
    }
    if ($missingTiers.Count -eq 0) {
        $tierDetail = '720P/1080P/2K/4K visible'
    } else {
        $tierDetail = 'missing: ' + ($missingTiers -join ',')
    }
    Add-Check -Name "$prefix.tier_cards" -Ok ($missingTiers.Count -eq 0) -Detail $tierDetail

    $saveAs = $Dom.Contains('另存为') -or $Dom.Contains('showSaveFilePicker') -or
        $Dom.Contains('下载') -or $lower.Contains('download')
    Add-Check -Name "$prefix.save_as" -Ok $saveAs -Detail 'save-as / download control present'

    $badgeText = ''
    foreach ($badge in @('AI 超分', '兼容模式')) {
        if ($Dom.Contains($badge)) { $badgeText = $badge; break }
    }
    if (-not $badgeText) {
        if ($lower.Contains('engine')) { $badgeText = '(engine keyword)' }
        elseif ($lower.Contains('badge')) { $badgeText = '(badge keyword)' }
    }
    Add-Check -Name "$prefix.engine_badge" -Ok ($badgeText -ne '') -Detail ("engine badge text: " + $badgeText) -Required $false

    # ---- R2: model selection cards, recommendation text, denoise controls ----
    $modelKeywords = @('智能推荐', 'Real-ESRGAN', 'Real-CUGAN', '动漫')
    $foundModels = @($modelKeywords | Where-Object { $Dom -match ('(?i)' + [regex]::Escape($_)) })
    $modelSection = $Dom.Contains('模型') -or $lower.Contains('model')
    Add-Check -Name "$prefix.model_section" -Ok $modelSection -Detail 'model selection section present'
    Add-Check -Name "$prefix.model_cards" -Ok ($foundModels.Count -ge 3) `
        -Detail ("model keywords found: " + ($foundModels -join ', '))
    $hasRecommend = $Dom.Contains('推荐') -or $lower.Contains('recommend')
    Add-Check -Name "$prefix.model_recommendation" -Ok $hasRecommend -Detail 'recommendation text present'
    $hasDenoiseText = $Dom.Contains('降噪')
    $hasDenoiseControl = $lower.Contains('type="checkbox"') -or $lower.Contains('type=''checkbox''') -or
        $Dom.Contains('降噪等级') -or $lower.Contains('<select') -or $lower.Contains('type="range"')
    Add-Check -Name "$prefix.denoise_controls" -Ok ($hasDenoiseText -and $hasDenoiseControl) `
        -Detail ("降噪 text={0}, control={1}" -f $hasDenoiseText, $hasDenoiseControl)
    Add-Check -Name "$prefix.model_used_hint" -Ok ($Dom.Contains('model_used') -or $lower.Contains('model used') -or
        $Dom.Contains('所用模型') -or $Dom.Contains('实际模型')) `
        -Detail 'result card shows model_used' -Required $false

    Add-Check -Name "$prefix.app_js" -Ok ($lower.Contains('app.js')) -Detail 'index references app.js' -Required $false
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
$startedAt = Get-Date
$browserResults = @()
$exitCode = 2
$anySuccess = $false

if ($browsers.Count -eq 0) {
    Write-Host '[WARN] no Edge/Chrome found; browser smoke test skipped'
    Add-Check -Name 'browser.available' -Ok $false -Detail 'no Edge/Chrome executable found' -Required $false
} else {
    Add-Check -Name 'browser.available' -Ok $true -Detail (($browsers | ForEach-Object { "$($_.Name):$($_.Path)" }) -join '; ')
    foreach ($browser in $browsers) {
        $tag = $browser.Name
        Write-Host "--- $tag $($browser.Path) ---"
        $domPath = Join-Path $ArtifactDir ("dom_{0}.html" -f $tag)
        $errPath = Join-Path $ArtifactDir ("browser_{0}.err.log" -f $tag)
        $code = Invoke-Headless -Exe $browser.Path -BrowserArguments @('--dump-dom') `
            -StdOutPath $domPath -StdErrPath $errPath
        $dom = ''
        if (Test-Path -LiteralPath $domPath) {
            $dom = Get-Content -LiteralPath $domPath -Raw -Encoding utf8
            if ($null -eq $dom) { $dom = '' }
        }
        if ($dom.Length -le 500) {
            Write-Host "[WARN] $tag --headless=new produced no DOM (exit $code); retrying with --headless"
            $fallbackArgs = @('--headless', '--dump-dom')
            $code = Invoke-Headless -Exe $browser.Path -BrowserArguments $fallbackArgs `
                -StdOutPath $domPath -StdErrPath $errPath
            if (Test-Path -LiteralPath $domPath) {
                $dom = Get-Content -LiteralPath $domPath -Raw -Encoding utf8
                if ($null -eq $dom) { $dom = '' }
            }
        }

        $shotPath = Join-Path $ArtifactDir ("screenshot_{0}.png" -f $tag)
        $shotOk = $false
        $shotBytes = 0
        if ($dom.Length -gt 500) {
            $anySuccess = $true
            Test-Dom -BrowserName $tag -Dom $dom
            $shotErrPath = Join-Path $ArtifactDir ("screenshot_{0}.err.log" -f $tag)
            Invoke-Headless -Exe $browser.Path -BrowserArguments @("--screenshot=$shotPath", '--window-size=1440,900') `
                -StdOutPath (Join-Path $ArtifactDir ("screenshot_{0}.out.log" -f $tag)) -StdErrPath $shotErrPath | Out-Null
            if (Test-Path -LiteralPath $shotPath) {
                $shotBytes = (Get-Item -LiteralPath $shotPath).Length
                $shotOk = $shotBytes -gt 10000
            }
            Add-Check -Name "browser.$tag.screenshot" -Ok $shotOk -Detail ("{0} bytes -> {1}" -f $shotBytes, $shotPath) -Required $false
        } else {
            Add-Check -Name "browser.$tag.dom_captured" -Ok $false -Detail "exit=$code, no DOM (see $errPath)"
        }

        $browserResults += [pscustomobject]@{
            name = $tag
            path = $browser.Path
            exit_code = $code
            dom_chars = $dom.Length
            dom_file = $domPath
            title = (Get-DomTitle -Dom $dom)
            tier_cards_visible = @('720P', '1080P', '2K', '4K' | Where-Object { $dom -match ('(?i)' + [regex]::Escape($_) + '\b') })
            save_as_present = ($dom.Contains('另存为') -or $dom.Contains('下载') -or $dom.Contains('showSaveFilePicker'))
            model_keywords = @('智能推荐', 'Real-ESRGAN', 'Real-CUGAN', '动漫' | Where-Object { $dom -match ('(?i)' + [regex]::Escape($_)) })
            model_section_present = ($dom.Contains('模型') -or $dom.ToLowerInvariant().Contains('model'))
            recommendation_present = ($dom.Contains('推荐') -or $dom.ToLowerInvariant().Contains('recommend'))
            denoise_present = ($dom.Contains('降噪'))
            screenshot = $shotPath
            screenshot_bytes = $shotBytes
        }
    }
    $requiredFail = @($script:Checks | Where-Object { $_.status -eq 'FAIL' }).Count
    if ($anySuccess) { $exitCode = if ($requiredFail -eq 0) { 0 } else { 1 } }
    else { $exitCode = 3 }
}

$pass = @($script:Checks | Where-Object { $_.status -eq 'PASS' }).Count
$failList = @($script:Checks | Where-Object { $_.status -eq 'FAIL' })
$warnList = @($script:Checks | Where-Object { $_.status -eq 'WARN' })
$payload = [pscustomobject]@{
    suite = 'browser-smoke'
    url = $Url
    started_at = $startedAt.ToString('o')
    browsers = $browserResults
    summary = [pscustomobject]@{
        pass = $pass; fail = $failList.Count; warn = $warnList.Count
        failed_names = @($failList | ForEach-Object { $_.name })
    }
    checks = @($script:Checks)
    artifacts = $ArtifactDir
}
$jsonPath = Join-Path $ArtifactDir 'browser_smoke.json'
$payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8
Write-Host "BROWSER SMOKE: browsers=$($browsers.Count) PASS=$pass FAIL=$($failList.Count) WARN=$($warnList.Count) -> exit $exitCode"
Write-Host "JSON: $jsonPath"
exit $exitCode
