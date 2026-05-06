# Usage: .\launch_27b.ps1              -> 27B model, FAST mode (128k)
#        .\launch_27b.ps1 -long       -> 27B model, LONG mode (256k)
#        .\launch_27b.ps1 -noRestart  -> run proxy once, no restart loop (debug)
#
# Launches proxy.py with the Qwen3.6-27B model. The proxy listens on port 8001
# and launches llama-server on port 8002 when traffic arrives.
#
# The 27B is a dense model (all 27B params active per token) vs the 35B-A3B
# which is MoE (~3B active). The 27B scores higher on coding/reasoning
# benchmarks but runs at similar speed on a 3090.

param(
    [switch]$long,
    [int]$idleTimeout = 600,
    [switch]$noRestart
)

$PythonCommand = if (Get-Command python -ErrorAction SilentlyContinue) {
    (Get-Command python -ErrorAction SilentlyContinue).Source
} else {
    Write-Error "python was not found in PATH. Install Python 3 and retry."
    exit 1
}

$ProxyScript = "H:\llama.cpp\proxy.py"
$args = @($ProxyScript, "--27b")
if ($long) { $args += "--long" }
$args += @("--idle-timeout", "$idleTimeout")
if ($noRestart) { $args += "--noRestart" }

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Qwen3.6-27B Proxy Launcher" -ForegroundColor Cyan
Write-Host "Model: Qwen3.6-27B (dense, Q4_K_XL)" -ForegroundColor Cyan
Write-Host "Mode: $(if ($long) {'LONG (256k)'} else {'FAST (128k)'})" -ForegroundColor Cyan
Write-Host "Proxy: 0.0.0.0:8001 -> 127.0.0.1:8002" -ForegroundColor Cyan
Write-Host "Idle timeout: ${idleTimeout}s" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

& $PythonCommand @args
