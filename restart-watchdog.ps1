# Graceful midnight restart for the chat watchdog.
# Closes the watchdog's console window (WM_CLOSE → CTRL_CLOSE_EVENT cascades
# through powershell → python → llama-server, all shut down via existing finally paths).
# Then launches a fresh watchdog in a new visible console.
#
# Intended as a daily Task Scheduler entry (see restart-watchdog-embed.ps1 for embed).

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path $PSCommandPath -Parent
$watchdog  = Join-Path $scriptDir "watchdog.ps1"

# Import shared timestamp helper
. (Join-Path $scriptDir "log_paths.ps1")
$logRoot   = Join-Path $scriptDir "logs"

function Log($msg) {
    $date = Get-Date -Format "yyyy-MM-dd"
    $logFile = Get-WeeklyLogPath -Root $logRoot -Name "watchdog-restart-$date.log"
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

# --- Find the watchdog console window and close it ---
$found = $false
$procs = Get-Process powershell -ErrorAction SilentlyContinue |
    Where-Object {
        $_.MainWindowTitle -like "*watchdog*" -and
        $_.MainWindowTitle -notlike "*restart*" -and
        $_.Id -ne $PID
    }

foreach ($proc in $procs) {
    if ($proc.MainWindowHandle -ne [IntPtr]::Zero) {
        Log "Closing watchdog window (PID $($proc.Id), title: '$($proc.MainWindowTitle)')"
        [Win32]::PostMessage($proc.MainWindowHandle, $WM_CLOSE, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
        $found = $true
    }
}

if (-not $found) {
    Log "No watchdog window found — nothing to close"
} else {
    # Wait for the process tree to finish shutting down.
    # Python's lifecycle_context finally block terminates llama-server (5s kill timeout),
    # so 15s gives plenty of margin for the full cascade.
    Log "Waiting up to 15s for process tree to exit..."
    $timeout = 15
    $elapsed = 0
    while ($elapsed -lt $timeout) {
        $stillRunning = Get-Process powershell -ErrorAction SilentlyContinue |
            Where-Object {
                $_.MainWindowTitle -like "*watchdog*" -and
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

# --- Launch fresh watchdog in a minimized console ---
Log "Starting fresh watchdog"
Start-Process powershell -WindowStyle Minimized `
    -ArgumentList "-NoExit", "-NoProfile", "-File", "`"$watchdog`"" `
    -WorkingDirectory $scriptDir

Log "Restart complete"
