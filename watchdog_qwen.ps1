# Supervisor for proxy.py — auto-restarts the proxy if it crashes.
# Usage: .\watchdog_qwen.ps1              -> Q3 model, FAST mode (128k)
#        .\watchdog_qwen.ps1 -q4          -> Q4 model, FAST mode (128k)
#        .\watchdog_qwen.ps1 -long        -> Q3 model, LONG mode (256k)
#        .\watchdog_qwen.ps1 -q4 -long    -> Q4 model, LONG mode (256k)
#        .\watchdog_qwen.ps1 -noRestart   -> run proxy once, no restart loop (debug)
#
# proxy.py owns the llama-server lifecycle. The proxy listens on port 8001 and
# launches llama-server on port 8002 when traffic arrives.

param(
    [switch]$long,
    [switch]$q4,
    [int]$idleTimeout = 600,
    [switch]$noRestart,
    [switch]$help
)

if ($help) {
    Write-Host "Usage: .\watchdog_qwen.ps1 [options]`
`
Options:
  -q4         Use Q4_K_M model (better quality, larger file)
  -long       Use 256k context window (slower, more VRAM)
    -idleTimeout  Idle timeout in seconds before llama-server is stopped
    -noRestart  Run proxy once, no auto-restart loop (for debugging)
  -help       Show this help message`
`
Examples:
  .\watchdog_qwen.ps1               # Q3 model, 128k context
  .\watchdog_qwen.ps1 -q4           # Q4 model, 128k context
  .\watchdog_qwen.ps1 -long         # Q3 model, 256k context
  .\watchdog_qwen.ps1 -q4 -long     # Q4 model, 256k context
    .\watchdog_qwen.ps1 -idleTimeout 60  # Stop backend after 1 idle minute
    .\watchdog_qwen.ps1 -noRestart    # Single run, no watchdog
  .\watchdog_qwen.ps1 -help         # Show this help
"; exit 0
}

if ($idleTimeout -lt 1) {
        Write-Error "idleTimeout must be at least 1 second."
        exit 1
}

$LogFile = "H:\llama.cpp\watchdog.log"
$ProxyScript = "H:\llama.cpp\proxy.py"
$MaxRestarts = 10          # max restarts within $RestartWindow seconds
$RestartWindow = 300       # 5-minute window for max restarts
$RestartDelay = 3          # seconds between restarts
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
Write-Log "Watchdog started (PID: $PID)"
$mode = if ($long) { 'LONG (256k)' } else { 'FAST (128k)' }
$model = if ($q4) { 'Q4_K_M' } else { 'Q3_K_XL' }
Write-Log "Model: $model | Mode: $mode"
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
        
        # Check if we're in a restart loop
        if ($RestartCount -gt $MaxRestarts) {
            Write-Log ('*** ERROR: Restart limit (' + $MaxRestarts + ' in ' + $RestartWindow + 's) reached. Stopping watchdog.')
            Write-Log "Check logs at: $LogFile"
            exit 1
        }
        
        Start-Sleep -Seconds $RestartDelay
    }

    Write-Log "--- Launching proxy.py (attempt #$RestartCount) ---"

    $args = @($ProxyScript)
    if ($q4) {
        $args += "--q4"
    }
    if ($long) {
        $args += "--long"
    }
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

    # Avoid rapid restart loops
    if ($RestartCount -ge $MaxRestarts) {
        $recentRestarts = 0
        # Simple throttle: if we've restarted too many times, increase delay
        $RestartDelay = [Math]::Min($RestartDelay * 2, 60)
        Write-Log ('Increasing restart delay to ' + $RestartDelay + 's to avoid rapid loops.')
    }
}
