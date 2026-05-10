# Supervisor for proxy.py — restarts the proxy if it crashes.
# Usage: .\watchdog.ps1 [extra args forwarded to proxy.py]
# Examples:
#   .\watchdog.ps1                                            # interactive: pick model + ctx
#   .\watchdog.ps1 --model "Qwen3.6-35B-A3B Q3" --ctx-size 32768  # headless
#   .\watchdog.ps1 --idle-timeout 600                         # production idle window

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path $PSCommandPath -Parent
$python    = Join-Path $scriptDir ".venv\Scripts\python.exe"
$proxy     = Join-Path $scriptDir "proxy.py"
$logFile   = Join-Path $scriptDir "watchdog.log"

if (-not (Test-Path $python)) { throw "Python not found at $python" }
if (-not (Test-Path $proxy))  { throw "proxy.py not found at $proxy" }

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

$delay = 3
$maxFastRestarts = 5
$fastWindow = 60   # seconds
$restarts = @()

Log "Watchdog starting (proxy=$proxy)"

while ($true) {
    $started = Get-Date
    Log "Launching proxy.py $($args -join ' ')"
    & $python $proxy @args
    $code = $LASTEXITCODE
    $duration = ((Get-Date) - $started).TotalSeconds
    Log ("proxy.py exited code={0} after {1:N1}s" -f $code, $duration)

    # Throttle: if too many crashes in a short window, back off
    $restarts = @($restarts | Where-Object { ((Get-Date) - $_).TotalSeconds -lt $fastWindow })
    $restarts += (Get-Date)
    if ($restarts.Count -ge $maxFastRestarts) {
        $delay = [Math]::Min($delay * 2, 60)
        Log "Crash loop detected ($($restarts.Count) restarts in ${fastWindow}s); backing off to ${delay}s"
        $restarts = @()
    }

    Start-Sleep -Seconds $delay
}
