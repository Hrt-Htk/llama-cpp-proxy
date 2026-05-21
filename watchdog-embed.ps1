# Supervisor for embed_proxy.py — restarts the embed proxy if it crashes.
# Usage: .\watchdog-embed.ps1 [extra args forwarded to embed_proxy.py]

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path $PSCommandPath -Parent
$python    = Join-Path $scriptDir ".venv\Scripts\python.exe"
$proxy     = Join-Path $scriptDir "embed_proxy.py"
$logRoot   = Join-Path $scriptDir "logs"

. (Join-Path $scriptDir "log_paths.ps1")

if (-not (Test-Path $python)) { throw "Python not found at $python" }
if (-not (Test-Path $proxy))  { throw "embed_proxy.py not found at $proxy" }

function Log($msg) {
    $date = Get-Date -Format "yyyy-MM-dd"
    $logFile = Get-WeeklyLogPath -Root $logRoot -Name "watchdog-embed-$date.log"
    $line = "[{0}] {1}" -f (Get-LocalTimestamp), $msg
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

$delay = 3
$maxFastRestarts = 5
$fastWindow = 60
$restarts = @()

Log "Embed watchdog starting (proxy=$proxy)"

while ($true) {
    $started = Get-Date
    Log "Launching embed_proxy.py $($args -join ' ')"
    & $python $proxy @args
    $code = $LASTEXITCODE
    $duration = ((Get-Date) - $started).TotalSeconds
    Log ("embed_proxy.py exited code={0} after {1:N1}s" -f $code, $duration)

    $restarts = @($restarts | Where-Object { ((Get-Date) - $_).TotalSeconds -lt $fastWindow })
    $restarts += (Get-Date)
    if ($restarts.Count -ge $maxFastRestarts) {
        $delay = [Math]::Min($delay * 2, 60)
        Log "Crash loop detected ($($restarts.Count) restarts in ${fastWindow}s); backing off to ${delay}s"
        $restarts = @()
    }

    Start-Sleep -Seconds $delay
}
