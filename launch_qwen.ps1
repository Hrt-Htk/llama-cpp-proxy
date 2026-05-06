# Usage: .\launch_qwen.ps1         -> fast mode (128k context, fully in VRAM)
#        .\launch_qwen.ps1 -long   -> long context mode (256k, q8_0 KV cache)
param([switch]$long)

if ($long) {
    $context = 262144
    $kvType  = "q8_0"
    Write-Host "Starting in LONG CONTEXT mode (256k, q8_0 KV cache)..."
} else {
    $context = 131072
    $kvType  = "q8_0"
    Write-Host "Starting in FAST mode (128k context, fully in VRAM)..."
}


& "H:\llama.cpp\llama.cpp_setup\llama-server.exe" `
  --model "H:\llama.cpp\models\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf" `
  --mmproj "H:\llama.cpp\models\mmproj-F16.gguf" `
  --alias "qwen3.6-35b-a3b" `
  -ngl 999 `
  --no-mmap `
  -fa on `
  --cache-type-k $kvType `
  --cache-type-v $kvType `
  -c $context `
  --jinja `
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 `
  --presence-penalty 0.0 `
  --port 8001 `
  --host 127.0.0.1 `
  --api-key rRZsSjRvaUuRMr5AeDA14rO9jaSlhSRhRtBI5ZlO
