<#
.SYNOPSIS
    Native (no-conda) installer for VAMToolbox on Windows + Python 3.13.

.DESCRIPTION
    Sets up everything needed to run VAMToolbox from a plain virtual environment:

      1. Locates a Python 3.13 interpreter.
      2. Creates a virtual environment (default: .\.venv).
      3. Downloads the standalone ASTRA Toolbox CUDA build for Python 3.13
         from https://astra-toolbox.com/downloads/ (the GitHub release zip),
         installs the bundled VC++ redistributable, and pip-installs the
         ASTRA wheel inside it.
      4. Installs all Python requirements (requirements-py313.txt).
      5. Installs VAMToolbox itself (editable) so `import vamtoolbox` works.
      6. Verifies the install (astra.use_cuda() and import vamtoolbox).

    No conda required. Re-run safely: an existing venv is reused.

.PARAMETER VenvPath
    Where to create the virtual environment. Default: .\.venv

.PARAMETER AstraVersion
    ASTRA Toolbox version to fetch. Default: 2.4.1

.PARAMETER AstraZip
    Path to an already-downloaded ASTRA Windows-Python zip (skips the download).

.PARAMETER SkipTorch
    Omit torch from the install (smaller; torch is only needed for the pyTorch
    ray-tracing / algebraic propagators, not for OSMO/BCLP + ASTRA).

.PARAMETER SkipVcRedist
    Do not run the bundled vc_redist.x64.exe (use if you already have the
    Visual Studio 2017+ x64 redistributable installed).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install.ps1 -SkipTorch -VenvPath .\env
#>
[CmdletBinding()]
param(
    [string]$VenvPath = ".\.venv",
    [string]$AstraVersion = "2.4.1",
    [string]$AstraZip = "",
    [switch]$SkipTorch,
    [switch]$SkipVcRedist
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
function Info($m) { Write-Host "[install] $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[install] $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "[install] ERROR: $m" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------------------
# 1. Locate Python 3.13
# ---------------------------------------------------------------------------
Info "Locating Python 3.13..."
$pyLauncher = $null
# Prefer the py launcher, which can target an exact version.
if (Get-Command py -ErrorAction SilentlyContinue) {
    try { & py -3.13 -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $pyLauncher = @("py", "-3.13") } } catch {}
}
if (-not $pyLauncher -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $v = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
    if ($v -eq "3.13") { $pyLauncher = @("python") }
}
if (-not $pyLauncher) {
    Die "Python 3.13 not found. Install it from https://www.python.org/downloads/ (or 'winget install Python.Python.3.13'), then re-run."
}
Info "Using Python: $($pyLauncher -join ' ')"

# ---------------------------------------------------------------------------
# 2. Create the virtual environment
# ---------------------------------------------------------------------------
$venvPy = Join-Path $VenvPath "Scripts\python.exe"
if (Test-Path $venvPy) {
    Info "Reusing existing venv at $VenvPath"
} else {
    Info "Creating venv at $VenvPath"
    $pyExe = $pyLauncher[0]
    $pyArgs = @(); if ($pyLauncher.Count -gt 1) { $pyArgs = $pyLauncher[1..($pyLauncher.Count - 1)] }
    & $pyExe @pyArgs -m venv $VenvPath
    if (-not (Test-Path $venvPy)) { Die "venv creation failed." }
}
Info "Upgrading pip / wheel..."
& $venvPy -m pip install --upgrade pip wheel | Out-Null

# ---------------------------------------------------------------------------
# 3. ASTRA Toolbox (standalone CUDA build)
# ---------------------------------------------------------------------------
$work = Join-Path $env:TEMP "vamtoolbox-astra-$AstraVersion"
New-Item -ItemType Directory -Force -Path $work | Out-Null

if (-not $AstraZip) {
    $zipName = "astra-toolbox-$AstraVersion-python313-win-x64.zip"
    $url = "https://github.com/astra-toolbox/astra-toolbox/releases/download/v$AstraVersion/$zipName"
    $AstraZip = Join-Path $work $zipName
    if (Test-Path $AstraZip) {
        Info "Using cached ASTRA download: $AstraZip"
    } else {
        Info "Downloading ASTRA $AstraVersion (CUDA build) ..."
        Info "  $url"
        try {
            Invoke-WebRequest -Uri $url -OutFile $AstraZip -UseBasicParsing
        } catch {
            Die "Failed to download ASTRA. Download '$zipName' manually from https://astra-toolbox.com/downloads/ and re-run with -AstraZip <path>. ($_)"
        }
    }
}
if (-not (Test-Path $AstraZip)) { Die "ASTRA zip not found: $AstraZip" }

$extract = Join-Path $work "extracted"
if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
Info "Extracting ASTRA ..."
Expand-Archive -Path $AstraZip -DestinationPath $extract -Force

$wheel = Get-ChildItem -Path $extract -Recurse -Filter "astra_toolbox-*.whl" | Select-Object -First 1
if (-not $wheel) { Die "No ASTRA .whl found inside $AstraZip" }

# Visual C++ 2017+ x64 redistributable (ASTRA's CUDA DLLs need it)
if (-not $SkipVcRedist) {
    $vc = Get-ChildItem -Path $extract -Recurse -Filter "vc_redist*.exe" | Select-Object -First 1
    if ($vc) {
        Info "Installing VC++ redistributable (quiet) ..."
        try {
            Start-Process -FilePath $vc.FullName -ArgumentList "/install","/quiet","/norestart" -Wait
        } catch {
            Warn "vc_redist install skipped/failed (may need admin, or already installed): $_"
        }
    }
}

Info "Installing ASTRA wheel: $($wheel.Name)"
& $venvPy -m pip install --force-reinstall --no-deps $wheel.FullName
if ($LASTEXITCODE -ne 0) { Die "pip install of the ASTRA wheel failed." }

# ---------------------------------------------------------------------------
# 4. Python requirements
# ---------------------------------------------------------------------------
$reqFile = Join-Path $root "requirements-py313.txt"
if (-not (Test-Path $reqFile)) { Die "requirements-py313.txt not found next to install.ps1" }

if ($SkipTorch) {
    Info "Installing requirements (without torch) ..."
    $tmpReq = Join-Path $work "requirements-no-torch.txt"
    Get-Content $reqFile | Where-Object { $_ -notmatch '^\s*torch\s*==' } | Set-Content -Encoding utf8 $tmpReq
    & $venvPy -m pip install -r $tmpReq
} else {
    Info "Installing requirements (this includes torch and can take a while) ..."
    & $venvPy -m pip install -r $reqFile
}
if ($LASTEXITCODE -ne 0) { Die "pip install of requirements failed." }

# ---------------------------------------------------------------------------
# 5. VAMToolbox itself (editable)
# ---------------------------------------------------------------------------
Info "Installing VAMToolbox (editable) ..."
& $venvPy -m pip install -e $root
if ($LASTEXITCODE -ne 0) { Die "pip install of VAMToolbox failed." }

# ---------------------------------------------------------------------------
# 6. Verify
# ---------------------------------------------------------------------------
Info "Verifying install ..."
$check = "import astra, vamtoolbox; print('astra', getattr(astra,'__version__','?'), 'CUDA:', astra.use_cuda()); print('vamtoolbox', vamtoolbox.__version__)"
$verify = & $venvPy -c $check 2>&1 | Out-String
Write-Host $verify
if ($LASTEXITCODE -ne 0) { Die "Verification failed (could not import astra/vamtoolbox)." }

Write-Host ""
Info "Done. VAMToolbox is installed in '$VenvPath'."
Write-Host "      Activate it with:  $VenvPath\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "      Then try:          python examples\voxelize_and_optimize.py" -ForegroundColor Green
if ($verify -match 'CUDA: False') {
    Warn "astra.use_cuda() is False - check that you have an NVIDIA GPU and a recent driver. CPU-only paths still work but are slow."
}
