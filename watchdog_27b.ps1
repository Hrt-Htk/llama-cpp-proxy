# Supervisor for proxy.py with Qwen3.6-27B model — auto-restarts the proxy if it crashes.
# Usage: .\watchdog_27b.ps1              -> 27B model, FAST mode (128k)
#        .\watchdog_27b.ps1 -long        -> 27B model, LONG mode (256k)
#        .\watchdog_27b.ps1 -noRestart   -> run proxy once, no restart loop (debug)
#
# proxy.py owns the llama-server lifecycle. The proxy listens on port 8001 and
# launches llama-server on port 8002 when traffic arrives.

param(
    [switch]$long,
    [int]$idleTimeout = 600,
    [switch]$noRestart,
    [switch]$help
)

if ($help) {
    Write-Host "Usage: .\watchdog_27b.ps1 [options]`
`
Options:
  -long       Use 256k context window (slower, more VRAM)
  -idleTimeout  Idle timeout in seconds before llama-server is stopped
  -noRestart  Run proxy once, no auto-restart loop (for debugging)
  -help       Show this help`
`
Examples:
  .\watchdog_27b.ps1               # 27B model, 128k context
  .\watchdog_27b.ps1 -long         # 27B model, 256k context
  .\watchdog_27b.ps1 -idleTimeout 60  # Stop backend after 1 idle minute
  .\watchdog_27b.ps1 -noRestart    # Single run, no watchdog
  .\watchdog_27b.ps1 -help         # Show this help
"; exit 0
}

if ($idleTimeout -lt 1) {
    Write-Error "idleTimeout must be at least 1 second."
    exit 1
}

$LogFile = "H:\llama.cpp\watchdog.log"
$ProxyScript = "H:\llama.cpp\proxy.py"
$MaxRestarts = 10
$RestartWindow = 300
$RestartDelay = 3
$RestartCount = 0
$StartTime = Get-Date

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $entry = "[$timestamp] $Message"
    Write-Host $entry
    Add-Content -Path $LogFile -Value $entry
}

$PythonCommand = if (Get-Command python -ErrorAction SilentlyContinue) {
    (Get-Command python -ErrorAction SilentlyContinue).Source
} else {
    Write-Error "python was not found in PATH. Install Python 3 and retry."
    exit 1
}

& $PythonCommand -c "import aiohttp" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Log "Installing missing Python dependency: aiohttp"
    & $PythonCommand -m pip install aiohttp --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Log "*** ERROR: Failed to install aiohttp for proxy.py"
        exit 1
    }
}

Write-Log "=========================================="
Write-Log "Watchdog 27B started (PID: $PID)"
$mode = if ($long) { 'LONG (256k)' } else { 'FAST (128k)' }
Write-Log "Model: Qwen3.6-27B (dense, Q4_K_XL) | Mode: $mode"
Write-Log "Python: $PythonCommand"
Write-Log "Proxy target: 0.0.0.0:8001 -> 127.0.0.1:8002"
Write-Log "Idle timeout: $idleTimeout seconds"
Write-Log "Max restarts: $MaxRestarts per $RestartWindow seconds"
Write-Log "=========================================="

while ($true) {
    $RestartCount++
    $Uptime = ((Get-Date) - $StartTime).TotalSeconds

    if ($RestartCount -gt 1) {
        Write-Log "Restart #$RestartCount (uptime of previous instance: $([math]::Round($Uptime, 1))s)..."

        if ($RestartCount -gt $MaxRestarts) {
            Write-Log ('*** ERROR: Restart limit (' + $MaxRestarts + ' in ' + $RestartWindow + 's) reached. Stopping watchdog.')
            Write-Log "Check logs at: $LogFile"
            exit 1
        }

        Start-Sleep -Seconds $RestartDelay
    }

    Write-Log "--- Launching proxy.py --27b (attempt #$RestartCount) ---"

    $args = @($ProxyScript, "--27b")
    if ($long) { $args += "--long" }
    $args += @("--idle-timeout", "$idleTimeout")

    & $PythonCommand @args
    $exitCode = $LASTEXITCODE
    $endTime = Get-Date
    $duration = ($endTime - $StartTime).TotalSeconds

    Write-Log "--- Proxy exited (code: $exitCode, duration: $([math]::Round($duration, 1))s) ---"

    if ($exitCode -ne 0) {
        Write-Log ('Exit code ' + $exitCode + ' detected — restarting in ' + $RestartDelay + 's...')
    } else {
        Write-Log "Clean exit (code 0). $noRestart`n"
        if ($noRestart) {
            Write-Log "Exiting (no-restart mode)."
            exit 0
        }
        Write-Log ('Restarting in ' + $RestartDelay + 's...')
    }

    if ($RestartCount -ge $MaxRestarts) {
        $RestartDelay = [Math]::Min($RestartDelay * 2, 60)
        Write-Log ('Increasing restart delay to ' + $RestartDelay + 's to avoid rapid loops.')
    }
}
