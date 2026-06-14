<#
Creates two daily Task Scheduler entries that restart the watchdogs at midnight.
MUST be run as Administrator (schtasks requires elevated privileges).
#>
$ErrorActionPreference = "Stop"
$scriptDir = Split-Path $PSCommandPath -Parent

Write-Host "Creating scheduled tasks for daily watchdog restart..."

# --- Chat watchdog restart at 00:00 ---
$chatArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$scriptDir\restart-watchdog.ps1`""
$chatResult = schtasks /Create `
    /TN "LLAMA-CPP Restart Chat Watchdog" `
    /TR "powershell.exe $chatArgs" `
    /ST 00:00 /SC DAILY /RL HIGHEST /F 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Chat watchdog task created." -ForegroundColor Green
} else {
    Write-Host "[FAIL] Chat watchdog task: $chatResult" -ForegroundColor Red
}

# --- Embed watchdog restart at 00:01 ---
$embedArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$scriptDir\restart-watchdog-embed.ps1`""
$embedResult = schtasks /Create `
    /TN "LLAMA-CPP Restart Embed Watchdog" `
    /TR "powershell.exe $embedArgs" `
    /ST 00:01 /SC DAILY /RL HIGHEST /F 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Embed watchdog task created." -ForegroundColor Green
} else {
    Write-Host "[FAIL] Embed watchdog task: $embedResult" -ForegroundColor Red
}

Write-Host ""
Write-Host "Verify with: schtasks /Query /TN 'LLAMA-CPP*'"
