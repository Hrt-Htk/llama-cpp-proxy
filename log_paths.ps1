# Shared log-path helpers for PowerShell wrappers (watchdogs + tunnel runner).
# Mirrors log_paths.py: writes into logs/<YYYY-WNN>/ buckets (local Europe/Zurich
# date), and prunes any week folder older than the two most recent.
#
# Dot-source from a script:
#     . "H:\llama.cpp\log_paths.ps1"
# Then call: Get-WeeklyLogPath -Root "H:\llama.cpp\logs" -Name "watchdog-2026-05-18.log"

function Get-LocalTimestamp {
    # 2026-05-18 19:37:42.123+02:00 — sortable, DST-aware, machine-parseable.
    Get-Date -Format "yyyy-MM-dd HH:mm:ss.fffzzz"
}

function Get-IsoWeekFolderName {
    # Matches Python's "%G-W%V" (ISO year + ISO week, zero-padded).
    $now = Get-Date
    $cal = [System.Globalization.CultureInfo]::InvariantCulture.Calendar
    $week = $cal.GetWeekOfYear(
        $now,
        [System.Globalization.CalendarWeekRule]::FirstFourDayWeek,
        [System.DayOfWeek]::Monday
    )
    # ISO year can differ from calendar year for early Jan / late Dec.
    $isoYear = $now.Year
    if ($now.Month -eq 1 -and $week -ge 52) { $isoYear -= 1 }
    elseif ($now.Month -eq 12 -and $week -eq 1) { $isoYear += 1 }
    return ('{0:D4}-W{1:D2}' -f $isoYear, $week)
}

function Get-WeeklyLogDir {
    param([Parameter(Mandatory)][string] $Root)
    $week = Get-IsoWeekFolderName
    $target = Join-Path $Root $week
    $isNew = -not (Test-Path $target)
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    if ($isNew) { Remove-OldWeekFolders -Root $Root -KeepLatest 2 }
    return $target
}

function Get-WeeklyLogPath {
    param(
        [Parameter(Mandatory)][string] $Root,
        [Parameter(Mandatory)][string] $Name
    )
    return Join-Path (Get-WeeklyLogDir -Root $Root) $Name
}

function Remove-OldWeekFolders {
    param(
        [Parameter(Mandatory)][string] $Root,
        [int] $KeepLatest = 2
    )
    $weeks = Get-ChildItem -Path $Root -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^\d{4}-W\d{2}$' } |
        Sort-Object Name
    $stale = $weeks | Select-Object -SkipLast $KeepLatest
    foreach ($d in $stale) {
        try {
            Remove-Item -Recurse -Force -LiteralPath $d.FullName
            Write-Host ("Pruned old log week folder: {0}" -f $d.Name)
        } catch {
            Write-Warning ("Could not prune {0}: {1}" -f $d.FullName, $_)
        }
    }
}
