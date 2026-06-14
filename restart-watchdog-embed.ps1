# Graceful midnight restart for the embed watchdog.
# Same pattern as restart-watchdog.ps1 but targets the embed watchdog window.
#
# Intended as a daily Task Scheduler entry staggered 1 minute after the chat restart.

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path $PSCommandPath -Parent
$watchdog  = Join-Path $scriptDir "watchdog-embed.ps1"

# Import shared timestamp helper
. (Join-Path $scriptDir "log_paths.ps1")
$logRoot   = Join-Path $scriptDir "logs"

function Log($msg) {
    $date = Get-Date -Format "yyyy-MM-dd"
    $logFile = Get-WeeklyLogPath -Root $logRoot -Name "watchdog-embed-restart-$date.log"
    $line = "[{0}] {1}" -f (Get-LocalTimestamp), $msg
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

# --- Import Win32 PostMessage to send WM_CLOSE ---
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class Win32 {
    [DllImport("user32.dll")]
    public static extern bool PostMessage(IntPtr hWnd, uint Msg, IntPtr WParam, IntPtr LParam);
}
'@ -Language CSharp

$WM_CLOSE = 0x10

# --- Find the embed watchdog console window and close it ---
$found = $false
$procs = Get-Process powershell -ErrorAction SilentlyContinue |
    Where-Object {
        $_.MainWindowTitle -like "*embed*" -and
        $_.MainWindowTitle -notlike "*restart*" -and
        $_.Id -ne $PID
    }

foreach ($proc in $procs) {
    if ($proc.MainWindowHandle -ne [IntPtr]::Zero) {
        Log "Closing embed watchdog window (PID $($proc.Id), title: '$($proc.MainWindowTitle)')"
        [Win32]::PostMessage($proc.MainWindowHandle, $WM_CLOSE, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
        $found = $true
    }
}

if (-not $found) {
    Log "No embed watchdog window found — nothing to close"
} else {
    Log "Waiting up to 15s for process tree to exit..."
    $timeout = 15
    $elapsed = 0
    while ($elapsed -lt $timeout) {
        $stillRunning = Get-Process powershell -ErrorAction SilentlyContinue |
            Where-Object {
                $_.MainWindowTitle -like "*embed*" -and
                $_.MainWindowTitle -notlike "*restart*" -and
                $_.Id -ne $PID
            }
        if (-not $stillRunning) { break }
        Start-Sleep -Seconds 1
        $elapsed++
    }
    Log "Process tree cleared"
}

# --- Brief pause before restart to let ports fully release ---
Start-Sleep -Seconds 3

# --- Launch fresh embed watchdog in a visible console ---
Log "Starting fresh embed watchdog"
Start-Process powershell -WindowStyle Normal `
    -ArgumentList "-NoExit", "-NoProfile", "-File", "`"$watchdog`"" `
    -WorkingDirectory $scriptDir

Log "Restart complete"
