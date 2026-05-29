#!/usr/bin/env bash
# tribev2-rs — remote CUDA rig (Windows host + WSL2 Ubuntu)
#
# Modelled on ../rlx/rig.sh: sync sources to a Windows machine, run benchmarks
# on native MSVC and inside WSL, compare RLX CUDA performance on both.
# tribev2-rs uses empty default Cargo features; rig builds pass --features explicitly.
#
# Quick start (from tribev2-rs on your Mac/Linux dev machine):
#
#   cp scripts/rig/local.env.example scripts/rig/local.env   # set RIG_HOST
#   ./rig.sh probe              # GPU + repo on Windows and WSL
#   ./rig.sh sync               # push this repo to the rig
#   ./rig.sh sync-data          # push data/model.safetensors (+ config)
#   ./rig.sh verify             # sync + doctor (both runtimes)
#   ./rig.sh --both bench-cuda  # encoder bench: CPU + CUDA on Win + WSL
#   ./rig.sh fetch-bench        # copy bench/rig/* back to this machine
#   ./rig.sh report-cuda        # print Windows vs WSL CUDA table
#
# Connection: scripts/rig/local.env, scripts/rig/.host, ../rlx/scripts/.host, or RIG_HOST

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIG_CONF_DIR="$ROOT/scripts/rig"

[[ -f "$RIG_CONF_DIR/local.env" ]] && source "$RIG_CONF_DIR/local.env"

_resolve_host() {
  if [[ -n "${RIG_HOST:-}" ]]; then
    printf '%s' "$RIG_HOST"
  elif [[ -f "$RIG_CONF_DIR/.host" ]]; then
    tr -d '[:space:]' < "$RIG_CONF_DIR/.host"
  elif [[ -f "$ROOT/../rlx/scripts/.host" ]]; then
    tr -d '[:space:]' < "$ROOT/../rlx/scripts/.host"
  else
    printf 'user@100.103.104.54'
  fi
}
HOST="$(_resolve_host)"

KEY="${SSH_KEY:-${RIG_SSH_KEY:-$HOME/.ssh/utm_windows}}"
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20)

RIG_WORKSPACE_WIN="${RIG_WORKSPACE_WIN:-D:/rlx-workspace}"
RIG_ROOT_WIN="${RIG_ROOT_WIN:-${RIG_WORKSPACE_WIN}/tribev2-rs}"
RIG_BUILD_WIN="${RIG_BUILD_WIN:-C:/Users/user/AppData/Local/rlx-workspace/tribev2-rs}"
RIG_BUILD_WORKSPACE_WIN="${RIG_BUILD_WORKSPACE_WIN:-C:/Users/user/AppData/Local/rlx-workspace}"
RIG_TARGET_WIN="${RIG_TARGET_WIN:-C:/Users/user/AppData/Local/tribev2-cargo-target}"
RUSTUP_WIN="${RIG_RUSTUP_WIN:-C:/Users/user/.cargo/bin/rustup.exe}"
TOOLCHAIN_BIN_WIN="${RIG_TOOLCHAIN_BIN_WIN:-C:/Users/user/.rustup/toolchains/stable-x86_64-pc-windows-msvc/bin}"
CARGO_WIN="${RIG_CARGO_WIN:-${TOOLCHAIN_BIN_WIN}/cargo.exe}"
RIG_WSL_DISTRO="${RIG_WSL_DISTRO:-Ubuntu}"
RIG_RUNTIME="${RIG_RUNTIME:-windows}"
RIG_RUNTIME_PINNED=0
RIG_BENCH_TAG="${RIG_BENCH_TAG:-$(date +%Y%m%d)}"

_rig_default_wsl_root() {
  local p="${RIG_ROOT_WIN}"
  if [[ "$p" =~ ^([A-Za-z]):/(.*)$ ]]; then
    local drive
    drive="$(printf '%s' "${BASH_REMATCH[1]}" | tr '[:upper:]' '[:lower:]')"
    printf '/mnt/%s/%s' "$drive" "${BASH_REMATCH[2]}"
  else
    printf '/mnt/d/rlx-workspace/tribev2-rs'
  fi
}
RIG_ROOT_WSL="${RIG_ROOT_WSL:-$(_rig_default_wsl_root)}"
RIG_DATA_WIN="${RIG_DATA_WIN:-${RIG_ROOT_WIN}/data}"

require_host() {
  [[ -n "$HOST" ]] || {
    echo "set RIG_HOST or scripts/rig/.host (see scripts/rig/local.env.example)" >&2
    exit 2
  }
}

ssh_rig() {
  require_host
  ssh "${SSH_OPTS[@]}" "$HOST" "$@"
}

psh() {
  require_host
  local enc script
  script="\$ProgressPreference='SilentlyContinue'; $1"
  enc="$(printf '%s' "$script" | iconv -t UTF-16LE | base64 | tr -d '\n')"
  ssh "${SSH_OPTS[@]}" "$HOST" \
    "powershell -NoProfile -NonInteractive -EncodedCommand $enc" \
    | grep -vE '^(#< CLIXML|<Objs)' || true
}

wsl_bash() {
  require_host
  local script="$1"
  ssh "${SSH_OPTS[@]}" "$HOST" \
    "wsl -d ${RIG_WSL_DISTRO} -e bash -ls" <<<"$script"
}

_rig_wsl_cuda_ld_snippet() {
  cat <<'EOF'
for _cuda_lib in /usr/local/cuda/lib64 /usr/local/cuda-12/lib64 /usr/local/cuda-12.6/lib64; do
  if [[ -f "${_cuda_lib}/libcublas.so" || -f "${_cuda_lib}/libcublas.so.12" ]]; then
    export LD_LIBRARY_PATH="${_cuda_lib}:${LD_LIBRARY_PATH:-}"
    break
  fi
done
unset _cuda_lib
EOF
}

_rig_env_bash() {
  cat <<EOF
export RLX_RIG_RUNTIME=wsl
if [[ -f "\${HOME}/rlx-workspace-mirror/tribev2-rs/Cargo.toml" ]]; then
  export RLX_RIG_ROOT="\${HOME}/rlx-workspace-mirror/tribev2-rs"
else
  export RLX_RIG_ROOT='$RIG_ROOT_WSL'
fi
export TRIBEV2_DATA_DIR="\${TRIBEV2_DATA_DIR:-\$RLX_RIG_ROOT/data}"
export PATH="\$HOME/.cargo/bin:\$PATH"
export RUSTUP_TOOLCHAIN="\${RIG_WSL_TOOLCHAIN:-stable}"
$(_rig_wsl_cuda_ld_snippet)
cd "\$RLX_RIG_ROOT" || exit 1
EOF
}

_rig_env_ps1() {
  cat <<EOF
\$env:RLX_RIG_RUNTIME = 'windows'
\$env:RLX_RIG_ROOT = '$RIG_BUILD_WIN'
\$dataC = '$RIG_BUILD_WIN/data'
\$dataD = '$RIG_DATA_WIN'
if (Test-Path "\$dataC/model.safetensors") { \$env:TRIBEV2_DATA_DIR = \$dataC }
else { \$env:TRIBEV2_DATA_DIR = \$dataD }
\$env:CARGO_TARGET_DIR = '$RIG_TARGET_WIN'
\$env:Path = '$TOOLCHAIN_BIN_WIN;C:/Users/user/.cargo/bin;' + \$env:Path
if (-not (Test-Path \$env:CARGO_TARGET_DIR)) {
  New-Item -ItemType Directory -Force -Path \$env:CARGO_TARGET_DIR | Out-Null
}
if (-not (Test-Path '$RIG_BUILD_WIN/Cargo.toml')) {
  Write-Error "missing Windows build tree at $RIG_BUILD_WIN — run: ./rig.sh sync"
  exit 1
}
Set-Location '$RIG_BUILD_WIN'
EOF
}

_rig_runtime_label() {
  case "$1" in
    windows) echo "windows ($RIG_BUILD_WIN)" ;;
    wsl) echo "wsl:${RIG_WSL_DISTRO} ($RIG_ROOT_WSL)" ;;
    *) echo "$1" ;;
  esac
}

_rig_runtimes() {
  case "$RIG_RUNTIME" in
    windows|wsl) echo "$RIG_RUNTIME" ;;
    both) echo "windows wsl" ;;
    *)
      echo "rig.sh: unknown RIG_RUNTIME=$RIG_RUNTIME" >&2
      exit 2
      ;;
  esac
}

_rig_each_runtime() {
  local kind="$1"
  shift
  local rt fail=0 fn
  for rt in $(_rig_runtimes); do
    echo "==> $(_rig_runtime_label "$rt")"
    fn="_cmd_${kind}_${rt}"
    if ! "$fn" "$@"; then
      fail=1
    fi
    echo
  done
  return "$fail"
}

usage() {
  sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  echo
  echo "--- resolved ---"
  echo "  HOST=${HOST}"
  echo "  SSH_KEY=${KEY}"
  echo "  RIG_RUNTIME=${RIG_RUNTIME}"
  echo "  Windows repo (WSL virtio): ${RIG_ROOT_WIN}"
  echo "  Windows build (MSVC):        ${RIG_BUILD_WIN}"
  echo "  Windows data:                ${RIG_DATA_WIN}"
  echo "  WSL repo:                    ${RIG_ROOT_WSL}"
}

# --- gpu ---
_cmd_gpu_windows() { ssh_rig 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv'; }
_cmd_gpu_wsl() { wsl_bash "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv"; }
cmd_gpu() {
  case "${1:-}" in windows|wsl|both) RIG_RUNTIME="$1"; shift ;; esac
  _rig_each_runtime gpu
}

# --- probe ---
_cmd_probe_windows() {
  ssh_rig "nvidia-smi -L" || echo "nvidia-smi: not available on Windows PATH"
  ssh_rig "%USERPROFILE%\\.cargo\\bin\\rustup.exe -V" 2>/dev/null || true
  local win="${RIG_ROOT_WIN//\//\\}"
  ssh_rig "if exist \"${win}\\Cargo.toml\" (echo repo=ok) else (echo repo=MISSING & exit /b 1)"
}

_cmd_probe_wsl() {
  wsl_bash "$(cat <<EOF
$(_rig_env_bash)
set -e
uname -sr
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo 'nvidia-smi: missing in WSL'
[[ -f Cargo.toml ]] && echo repo=ok || { echo repo=MISSING; exit 1; }
[[ -f data/model.safetensors ]] && echo weights=ok || echo 'weights=missing (./rig.sh sync-data)'
command -v cargo >/dev/null && cargo -V || echo 'cargo=missing (./rig.sh setup-wsl-rust)'
EOF
)"
}

cmd_probe() {
  case "${1:-}" in windows|wsl|both) RIG_RUNTIME="$1"; RIG_RUNTIME_PINNED=1; shift ;; esac
  [[ "$RIG_RUNTIME_PINNED" -eq 0 ]] && RIG_RUNTIME=both
  echo "==> tribev2 rig $HOST (runtime=$RIG_RUNTIME)"
  _rig_each_runtime probe
}

# --- doctor ---
_cmd_doctor_windows() {
  local check="${1:-all}"
  psh "
\$fail = 0
function Ok(\$m) { Write-Host \"[ok] \$m\" }
function Bad(\$m) { Write-Host \"[fail] \$m\"; \$script:fail++ }
if ('$check' -eq 'all' -or '$check' -eq 'paths') {
  foreach (\$p in @('$RIG_ROOT_WIN', '$RIG_BUILD_WIN', '$RIG_DATA_WIN')) {
    if (Test-Path \$p) { Ok \"path \$p\" } else { Bad \"missing \$p\" }
  }
  if (Test-Path '$RIG_BUILD_WIN/Cargo.toml') { Ok 'Cargo.toml (C: build)' } else { Bad 'Cargo.toml — run sync' }
  if (Test-Path '$RIG_DATA_WIN/model.safetensors') { Ok 'model.safetensors' } else { Bad 'weights — run sync-data' }
}
if ('$check' -eq 'all' -or '$check' -eq 'gpu') {
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    \$g = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    if (\$LASTEXITCODE -eq 0) { Ok \"gpu \$g\" } else { Bad \"nvidia-smi: \$g\" }
  } else { Bad 'nvidia-smi not in PATH' }
}
if ('$check' -eq 'all' -or '$check' -eq 'cargo') {
  Set-Location '$RIG_BUILD_WIN'
  \$env:CARGO_TARGET_DIR = '$RIG_TARGET_WIN'
  \$out = & '$CARGO_WIN' -V 2>&1
  if (\$LASTEXITCODE -eq 0) { Ok \"cargo \$out\" } else { Bad \"cargo: \$out\" }
}
if (\$fail -gt 0) { exit 1 }
Write-Host 'doctor: windows passed'
"
}

_cmd_doctor_wsl() {
  local check="${1:-all}"
  wsl_bash "$(cat <<EOF
$(_rig_env_bash)
set -e
fail=0
ok() { echo "[ok] \$1"; }
bad() { echo "[fail] \$1"; fail=1; }
check='$check'
[[ -d "\$RLX_RIG_ROOT" ]] && ok "path \$RLX_RIG_ROOT" || bad "missing repo"
[[ -f Cargo.toml ]] && ok Cargo.toml || bad Cargo.toml
[[ -f "\$TRIBEV2_DATA_DIR/model.safetensors" ]] && ok weights || bad 'weights missing — sync-data'
if [[ "\$check" == all || "\$check" == gpu ]]; then
  if command -v nvidia-smi >/dev/null; then
    g=\$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader) && ok "gpu \$g" || bad "nvidia-smi failed"
  else bad 'nvidia-smi missing in WSL'
  fi
fi
if [[ "\$check" == all || "\$check" == cargo ]]; then
  command -v cargo >/dev/null && ok "cargo \$(cargo -V)" || bad 'cargo missing — setup-wsl-rust'
fi
$(_rig_wsl_cuda_ld_snippet)
for d in /usr/local/cuda/lib64 /usr/local/cuda-12/lib64; do
  [[ -f "\$d/libcublas.so" || -f "\$d/libcublas.so.12" ]] && ok "libcublas in \$d" && break
done
[[ \$fail -eq 0 ]] || exit 1
echo 'doctor: wsl passed'
EOF
)"
}

cmd_doctor() {
  case "${1:-}" in windows|wsl|both) RIG_RUNTIME="$1"; RIG_RUNTIME_PINNED=1; shift ;; esac
  [[ "$RIG_RUNTIME_PINNED" -eq 0 ]] && RIG_RUNTIME=both
  _rig_each_runtime doctor "${1:-all}"
}

_rig_mirror_win_tree() {
  local src="$1" dst="$2"
  psh "
if (-not (Test-Path '$src')) { Write-Error 'missing $src'; exit 1 }
if (-not (Test-Path '$dst')) { New-Item -ItemType Directory -Force -Path '$dst' | Out-Null }
& robocopy '$src' '$dst' /MIR /XD target .git data /NFL /NDL /NJH /NJS /NC /NS /NP
\$rc = \$LASTEXITCODE
if (\$rc -ge 8) { exit \$rc }
exit 0
"
}

cmd_sync() {
  require_host
  [[ -f "$ROOT/Cargo.toml" ]] || { echo "missing workspace: $ROOT" >&2; exit 1; }
  echo "==> sync tribev2-rs -> $HOST:$RIG_ROOT_WIN"
  psh "if (-not (Test-Path '$RIG_ROOT_WIN')) { New-Item -ItemType Directory -Force -Path '$RIG_ROOT_WIN' | Out-Null }"
  local -a excludes=(
    --exclude=.git --exclude=target --exclude=data
    --exclude=.DS_Store --exclude='._*' --exclude='.venv*' --exclude=venv --exclude=node_modules
  )
  COPYFILE_DISABLE=1 tar czf - -C "$ROOT" "${excludes[@]}" . \
    | ssh "${SSH_OPTS[@]}" "$HOST" \
        "powershell -NoProfile -Command \"cd '$RIG_ROOT_WIN'; tar xzf -\""
  echo "==> mirror -> $RIG_BUILD_WIN (MSVC build tree)"
  _rig_mirror_win_tree "$RIG_ROOT_WIN" "$RIG_BUILD_WIN"
  echo "==> synced (WSL: $RIG_ROOT_WSL)"
}

cmd_sync_data() {
  require_host
  local src="${TRIBEV2_DATA_SRC:-$ROOT/data}"
  [[ -f "$src/model.safetensors" ]] || {
    echo "sync-data: missing $src/model.safetensors" >&2
    exit 1
  }
  echo "==> sync data/ -> $HOST:$RIG_DATA_WIN"
  psh "if (-not (Test-Path '$RIG_DATA_WIN')) { New-Item -ItemType Directory -Force -Path '$RIG_DATA_WIN' | Out-Null }"
  local -a files=(model.safetensors config.yaml build_args.json)
  for f in "${files[@]}"; do
    [[ -f "$src/$f" ]] || continue
    echo "  $f"
    scp "${SSH_OPTS[@]}" "$src/$f" "$HOST:$RIG_DATA_WIN/"
  done
  if [[ -d "$src/parity_refs" ]]; then
    echo "  parity_refs/ (subset)"
    psh "if (-not (Test-Path '$RIG_DATA_WIN/parity_refs')) { New-Item -ItemType Directory -Force -Path '$RIG_DATA_WIN/parity_refs' | Out-Null }"
    for f in "$src/parity_refs"/*.bin; do
      [[ -f "$f" ]] || continue
      scp "${SSH_OPTS[@]}" "$f" "$HOST:$RIG_DATA_WIN/parity_refs/" 2>/dev/null || true
    done
  fi
  psh "
if (Test-Path '$RIG_BUILD_WIN') {
  if (-not (Test-Path '$RIG_DATA_WIN')) { exit 0 }
  \$d = '$RIG_BUILD_WIN/data'
  if (-not (Test-Path \$d)) { New-Item -ItemType Directory -Force -Path \$d | Out-Null }
  Copy-Item -Force '$RIG_DATA_WIN/model.safetensors' \"\$d/model.safetensors\" -ErrorAction SilentlyContinue
  Copy-Item -Force '$RIG_DATA_WIN/config.yaml' \"\$d/config.yaml\" -ErrorAction SilentlyContinue
  Copy-Item -Force '$RIG_DATA_WIN/build_args.json' \"\$d/build_args.json\" -ErrorAction SilentlyContinue
}
"
  echo "==> data synced"
}

_rig_wsl_sync_workspace_mirror() {
  echo "==> WSL mirror virtio tree -> ~/rlx-workspace-mirror (ext4, avoids rustc ICE on /mnt/d)"
  wsl_bash "$(cat <<EOF
set -e
_ws_src='$RIG_ROOT_WSL'
_ws_mirror="\${HOME}/rlx-workspace-mirror"
if [[ "\$_ws_src" != /mnt/* ]]; then
  echo "skip mirror (not virtio): \$_ws_src"
  exit 0
fi
mkdir -p "\$_ws_mirror"
rsync -a --delete "\${_ws_src}/" "\${_ws_mirror}/" \
  --exclude target --exclude .git --exclude '._*' --exclude .DS_Store
echo "mirror ok"
EOF
)"
}

_cmd_bench_cuda_windows() {
  psh "
$(_rig_env_ps1)
\$env:RIG_BENCH_TAG = '${RIG_BENCH_TAG}_windows'
\$bash = 'C:/Program Files/Git/bin/bash.exe'
if (-not (Test-Path \$bash)) {
  \$cmd = Get-Command bash -ErrorAction SilentlyContinue
  if (\$cmd) { \$bash = \$cmd.Source } else { Write-Error 'Git Bash required (install Git for Windows)'; exit 1 }
}
& \$bash './bench/run_cuda_rig.sh'
"
}

_cmd_bench_cuda_wsl() {
  _rig_wsl_sync_workspace_mirror || true
  wsl_bash "$(cat <<EOF
$(_rig_env_bash)
export RIG_BENCH_TAG='${RIG_BENCH_TAG}_wsl'
chmod +x bench/run_cuda_rig.sh 2>/dev/null || true
./bench/run_cuda_rig.sh
EOF
)"
}

cmd_bench_cuda() {
  case "${1:-}" in windows|wsl|both) RIG_RUNTIME="$1"; RIG_RUNTIME_PINNED=1; shift ;; esac
  [[ "$RIG_RUNTIME_PINNED" -eq 0 ]] && RIG_RUNTIME=both
  echo "==> bench-cuda tag=${RIG_BENCH_TAG} runtime=$RIG_RUNTIME"
  _rig_each_runtime bench_cuda
}

cmd_bench() {
  case "${1:-}" in windows|wsl|both) RIG_RUNTIME="$1"; RIG_RUNTIME_PINNED=1; shift ;; esac
  _rig_run_bench_script() {
    local rt="$1"
    case "$rt" in
      windows)
        psh "
$(_rig_env_ps1)
& 'C:/Program Files/Git/bin/bash.exe' -lc './bench/run_all_backends.sh'
" || psh "$(_rig_env_ps1); & bash './bench/run_all_backends.sh'"
        ;;
      wsl)
        _rig_wsl_sync_workspace_mirror || true
        wsl_bash "$(cat <<EOF
$(_rig_env_bash)
chmod +x bench/run_all_backends.sh
./bench/run_all_backends.sh
EOF
)"
        ;;
    esac
  }
  for rt in $(_rig_runtimes); do
    echo "==> $(_rig_runtime_label "$rt") full backend sweep"
    _rig_run_bench_script "$rt" || true
    echo
  done
}

cmd_fetch_bench() {
  require_host
  mkdir -p "$ROOT/bench/rig"
  local win_glob="${RIG_BUILD_WIN}/bench/rig"
  echo "==> fetch Windows ${win_glob}/*"
  psh "
\$src = '$win_glob'
if (-not (Test-Path \$src)) { Write-Host '  (no Windows results)'; exit 0 }
Get-ChildItem -Path \$src -Directory | ForEach-Object { Write-Host \$_.FullName }
" || true
  # Tar stream avoids Windows path escaping issues.
  ssh "${SSH_OPTS[@]}" "$HOST" \
    "powershell -NoProfile -Command \"if (Test-Path '$win_glob') { Set-Location '$win_glob'; tar czf - . }\"" \
    | tar xzf - -C "$ROOT/bench/rig" 2>/dev/null || echo "  (no Windows bench/rig yet)"

  echo "==> fetch WSL bench/rig (via tar in WSL)"
  wsl_bash "$(cat <<EOF
set -e
$(_rig_env_bash)
d="\$RLX_RIG_ROOT/bench/rig"
if [[ -d "\$d" ]]; then cd "\$d" && tar czf - .; else exit 1; fi
EOF
)" | tar xzf - -C "$ROOT/bench/rig" 2>/dev/null || echo "  (no WSL bench/rig yet)"

  echo "==> local bench/rig:"
  find "$ROOT/bench/rig" -maxdepth 2 -type f -name 'results_*.json' 2>/dev/null | head -20 || true
}

cmd_report_cuda() {
  if [[ ! -d "$ROOT/bench/rig" ]] || [[ -z "$(ls -A "$ROOT/bench/rig" 2>/dev/null)" ]]; then
    echo "No local results — run: ./rig.sh fetch-bench" >&2
    exit 1
  fi
  python3 "$ROOT/scripts/rig/print_cuda_report.py" "${RIG_BENCH_TAG}"
}

cmd_setup_wsl_rust() {
  echo "==> install rustup in WSL (${RIG_WSL_DISTRO})"
  wsl_bash "$(cat <<'EOF'
set -e
export PATH="$HOME/.cargo/bin:$PATH"
if command -v cargo >/dev/null; then cargo -V; exit 0; fi
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup default stable
cargo -V
EOF
)"
}

cmd_setup_wsl_cuda() {
  echo "==> hint: install NVIDIA CUDA toolkit inside WSL (apt) for libcublas"
  echo "  https://docs.nvidia.com/cuda/wsl-user-guide/index.html"
  ssh_rig "wsl -d ${RIG_WSL_DISTRO} -u root bash -ls" <<'EOF'
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential pkg-config libssl-dev rsync || true
EOF
}

_args_to_b64() {
  local args_json
  args_json="$(printf '%s\0' "$@" | python3 -c 'import json,sys; print(json.dumps([a for a in sys.stdin.read().split("\x00") if a]))')"
  printf '%s' "$args_json" | base64 | tr -d '\n'
}

_cmd_run_windows() {
  local args_b64="$1"
  psh "
$(_rig_env_ps1)
\$a = [string[]]([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$args_b64')) | ConvertFrom-Json)
& \$a[0] \$a[1..(\$a.Length-1)]
"
}

_cmd_run_wsl() {
  local args_b64="$1"
  wsl_bash "$(cat <<EOF
$(_rig_env_bash)
python3 -c "
import base64, json, os, subprocess
os.chdir(os.environ['RLX_RIG_ROOT'])
args = json.loads(base64.b64decode('${args_b64}'))
subprocess.check_call(args)
"
EOF
)"
}

cmd_run() {
  [[ "${1:-}" == "--" ]] && shift
  [[ $# -ge 1 ]] || { echo "run: pass command after --" >&2; exit 2; }
  local args_b64
  args_b64="$(_args_to_b64 "$@")"
  case "$RIG_RUNTIME" in
    windows) _cmd_run_windows "$args_b64" ;;
    wsl) _cmd_run_wsl "$args_b64" ;;
    both)
      _cmd_run_windows "$args_b64" || return 1
      echo
      _cmd_run_wsl "$args_b64"
      ;;
  esac
}

cmd_cargo() {
  [[ $# -ge 1 ]] || { echo "cargo: pass args" >&2; exit 2; }
  case "$RIG_RUNTIME" in
    windows) cmd_run -- "$CARGO_WIN" "$@" ;;
    wsl) cmd_run -- cargo "$@" ;;
    both)
      RIG_RUNTIME=windows
      cmd_run -- "$CARGO_WIN" "$@" || win_fail=1
      echo
      RIG_RUNTIME=wsl
      cmd_run -- cargo "$@" || wsl_fail=1
      [[ "${win_fail:-0}" -eq 0 && "${wsl_fail:-0}" -eq 0 ]]
      ;;
  esac
}

cmd_test_cuda() {
  local feat="pure-rust,rlx-encoder,rlx-cpu,rlx-cuda-enc"
  echo "==> tribev2 RLX CUDA parity (encoder-only features)"
  if [[ "$(_rig_runtimes)" == *wsl* ]]; then
    _rig_wsl_sync_workspace_mirror || true
  fi
  cmd_cargo test -p tribev2 --release --no-default-features --features "$feat" \
    --test rlx_parity -- --nocapture test_rlx_vs_pure_rust_on_cuda_device || true
}

cmd_verify() {
  cmd_sync
  RIG_RUNTIME=both
  cmd_doctor all
}

cmd_ssh() {
  require_host
  if [[ "${1:-}" == "--wsl" ]]; then
    shift
    exec ssh "${SSH_OPTS[@]}" -t "$HOST" "wsl -d ${RIG_WSL_DISTRO}"
  fi
  exec ssh "${SSH_OPTS[@]}" -t "$HOST" "$@"
}

main() {
  local -a argv=("$@")
  set --
  local item
  for item in "${argv[@]}"; do
    case "$item" in
      --windows) RIG_RUNTIME=windows; RIG_RUNTIME_PINNED=1 ;;
      --wsl) RIG_RUNTIME=wsl; RIG_RUNTIME_PINNED=1 ;;
      --both) RIG_RUNTIME=both; RIG_RUNTIME_PINNED=1 ;;
      *) set -- "$@" "$item" ;;
    esac
  done

  local action="${1:-help}"
  shift || true

  case "$action" in
    help|-h|--help) usage ;;
    gpu) cmd_gpu "$@" ;;
    probe|status) cmd_probe "$@" ;;
    doctor|check) cmd_doctor "$@" ;;
    verify) cmd_verify "$@" ;;
    sync|push) cmd_sync "$@" ;;
    sync-data) cmd_sync_data "$@" ;;
    setup-wsl-rust) cmd_setup_wsl_rust "$@" ;;
    setup-wsl-cuda) cmd_setup_wsl_cuda "$@" ;;
    bench-cuda) cmd_bench_cuda "$@" ;;
    bench) cmd_bench "$@" ;;
    fetch-bench) cmd_fetch_bench "$@" ;;
    report-cuda|report) cmd_report_cuda "$@" ;;
    test|test-cuda) cmd_test_cuda "$@" ;;
    run) cmd_run "$@" ;;
    cargo) cmd_cargo "$@" ;;
    ssh|shell) cmd_ssh "$@" ;;
    exec) require_host; ssh_rig "$@" ;;
    *)
      echo "rig.sh: unknown command: $action" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
