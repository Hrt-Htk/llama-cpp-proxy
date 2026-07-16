# Supervisor for proxy.py — restarts the proxy if it crashes.
# Usage: .\watchdog.ps1 [extra args forwarded to proxy.py]
# Examples:
#   .\watchdog.ps1                                            # uses defaults
#   .\watchdog.ps1 --model "Qwen3.6-35B-A3B Q3" --ctx-size 32768  # override default fallback
#   .\watchdog.ps1 --idle-timeout 600                         # production idle window
# All (model x ctx) combos are exposed as router presets regardless;
# --model and --ctx-size only set the fallback when a client omits the model.

$ErrorActionPreference = "Stop"

# Stable console title so the daily restart scripts can find this window by
# title (matches *watchdog*, and deliberately contains no "embed" so the chat
# restart never catches the embed watchdog). Also survives minimized launches.
$host.UI.RawUI.WindowTitle = "llama-chat-watchdog"

$scriptDir = Split-Path $PSCommandPath -Parent
$python    = Join-Path $scriptDir ".venv\Scripts\python.exe"
$proxy     = Join-Path $scriptDir "proxy.py"
$logRoot   = Join-Path $scriptDir "logs"

. (Join-Path $scriptDir "log_paths.ps1")

if (-not (Test-Path $python)) { throw "Python not found at $python" }
if (-not (Test-Path $proxy))  { throw "proxy.py not found at $proxy" }

function Log($msg) {
    $date = Get-Date -Format "yyyy-MM-dd"
    $logFile = Get-WeeklyLogPath -Root $logRoot -Name "watchdog-$date.log"
    $line = "[{0}] {1}" -f (Get-LocalTimestamp), $msg
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
