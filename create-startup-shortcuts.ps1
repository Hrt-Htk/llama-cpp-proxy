<#
Installs (or removes) startup shortcuts so both watchdogs launch automatically,
minimized to the taskbar, when you log in.

- Runs at LOGON in your interactive desktop session (fires on Windows auto-login,
  so no username/password is required — unlike a "run whether logged on or not"
  boot task, which would need stored credentials).
- Non-elevated, matching how the watchdogs are normally run by hand. This also
  avoids the elevated-orphan-worker problem seen with /RL HIGHEST tasks.
- Minimized window (not hidden) so the daily restart scripts can still find the
  console by title and send WM_CLOSE for a clean llama-server shutdown.

Usage:
    .\create-startup-shortcuts.ps1            # install
    .\create-startup-shortcuts.ps1 -Remove    # uninstall
#>
param([switch]$Remove)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path $PSCommandPath -Parent
$startup   = [Environment]::GetFolderPath('Startup')

# WindowStyle for .lnk: 1 = Normal, 3 = Maximized, 7 = Minimized
$MINIMIZED = 7

$shortcuts = @(
    @{ Name = "LLAMA Chat Watchdog";  Script = "watchdog.ps1" }
    @{ Name = "LLAMA Embed Watchdog"; Script = "watchdog-embed.ps1" }
)

if ($Remove) {
    foreach ($s in $shortcuts) {
        $lnk = Join-Path $startup "$($s.Name).lnk"
        if (Test-Path $lnk) {
            Remove-Item $lnk -Force
            Write-Host "[OK] Removed $lnk" -ForegroundColor Green
        } else {
            Write-Host "[--] Not present: $lnk" -ForegroundColor DarkGray
        }
    }
    return
}

$wsh = New-Object -ComObject WScript.Shell

foreach ($s in $shortcuts) {
    $target = Join-Path $scriptDir $s.Script
    if (-not (Test-Path $target)) { throw "Watchdog script not found: $target" }

    $lnkPath = Join-Path $startup "$($s.Name).lnk"
    $lnk = $wsh.CreateShortcut($lnkPath)
    $lnk.TargetPath       = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $lnk.Arguments        = "-NoProfile -ExecutionPolicy Bypass -NoExit -File `"$target`""
    $lnk.WorkingDirectory = $scriptDir
    $lnk.WindowStyle      = $MINIMIZED
    $lnk.Description      = "Auto-start $($s.Script) minimized at logon"
    $lnk.Save()
    Write-Host "[OK] Installed $lnkPath (minimized, launches $($s.Script))" -ForegroundColor Green
}

Write-Host ""
Write-Host "Startup folder: $startup"
Write-Host "They will launch minimized on your next login. To start them now:"
Write-Host "  Start-Process powershell -WindowStyle Minimized -ArgumentList '-NoProfile','-NoExit','-File','$(Join-Path $scriptDir "watchdog.ps1")'"
Write-Host "  Start-Process powershell -WindowStyle Minimized -ArgumentList '-NoProfile','-NoExit','-File','$(Join-Path $scriptDir "watchdog-embed.ps1")'"
Write-Host ""
Write-Host "To remove: .\create-startup-shortcuts.ps1 -Remove"
