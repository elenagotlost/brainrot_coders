# Start the board's two model servers on a Windows laptop, with the SAME flags as
# board/systemd/math-llm.service and math-embed.service, so laptop answers match the board.
#
#   powershell -File laptop\serve_windows.ps1            # start both servers
#   powershell -File laptop\serve_windows.ps1 -Stop      # stop them
#   powershell -File laptop\serve_windows.ps1 -Model "$HOME\models\Qwen_Qwen3.5-2B-Q4_K_M.gguf"
#
# Needs: llama.cpp Windows CPU build (b12311, closest release to the pinned commit)
# unzipped in ~\llama.cpp, and the models from config.env in ~\models.
param(
    [string]$Model = "",
    [switch]$Stop
)
$ErrorActionPreference = "Stop"

if ($Stop) {
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
    Write-Host "Stopped llama-server."
    exit 0
}

# Read config.env (KEY=VALUE, same file the board scripts use)
$root = Split-Path -Parent $PSScriptRoot
$cfg = @{}
foreach ($line in Get-Content (Join-Path $root "config.env")) {
    $line = ($line -split "#", 2)[0].Trim()
    if ($line -match '^([A-Z_]+)=(.*)$') { $cfg[$Matches[1]] = $Matches[2].Trim('"') }
}

$server = Join-Path $HOME "llama.cpp\llama-server.exe"
$models = Join-Path $HOME "models"
if (-not $Model) { $Model = Join-Path $models $cfg.LLM_FILE }
$embed = Join-Path $models $cfg.EMBED_FILE
$logs = Join-Path $root "results"
New-Item -ItemType Directory -Force $logs | Out-Null

foreach ($f in @($server, $Model, $embed)) {
    if (-not (Test-Path $f)) { throw "Missing: $f" }
}
foreach ($port in @($cfg.LLM_PORT, $cfg.EMBED_PORT)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $port is already in use (another llama-server, or 'adb forward' to the board). Run 'adb forward --remove-all' or -Stop first."
    }
}

$alias = if ($Model -like "*$($cfg.LLM_FILE)") { $cfg.LLM_ALIAS } else { [IO.Path]::GetFileNameWithoutExtension($Model) }

$llmArgs = @("-m", $Model, "--host", "127.0.0.1", "--port", $cfg.LLM_PORT,
    "--threads", $cfg.LLM_THREADS, "--ctx-size", $cfg.LLM_CTX, "--parallel", "1",
    "--cache-ram", "0", "--ctx-checkpoints", "2", "--jinja",
    "--reasoning", "off", "--reasoning-budget", "0", "--alias", $alias)
$embedArgs = @("-m", $embed, "--embedding", "--host", "127.0.0.1", "--port", $cfg.EMBED_PORT,
    "--threads", $cfg.EMBED_THREADS, "--ctx-size", $cfg.EMBED_CTX, "--parallel", "1",
    "--cache-ram", "0", "--batch-size", $cfg.EMBED_CTX, "--ubatch-size", $cfg.EMBED_CTX)

Start-Process $server -ArgumentList $llmArgs -WindowStyle Hidden `
    -RedirectStandardError (Join-Path $logs "llm-server.log") -RedirectStandardOutput (Join-Path $logs "llm-server.out")
Start-Process $server -ArgumentList $embedArgs -WindowStyle Hidden `
    -RedirectStandardError (Join-Path $logs "embed-server.log") -RedirectStandardOutput (Join-Path $logs "embed-server.out")

Write-Host "Waiting for the servers..."
foreach ($port in @($cfg.LLM_PORT, $cfg.EMBED_PORT)) {
    $ok = $false
    for ($i = 0; $i -lt 60 -and -not $ok; $i++) {
        try { $ok = (Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 2).status -eq "ok" } catch { Start-Sleep 1 }
    }
    if ($ok) { Write-Host "  port $port up" } else { throw "Server on port $port did not start: see $logs\*-server.log" }
}
Write-Host "Language model ($alias): http://127.0.0.1:$($cfg.LLM_PORT)   Embeddings: port $($cfg.EMBED_PORT)"
