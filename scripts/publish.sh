#!/usr/bin/env bash
# publish.sh — publish tribev2-rs workspace crates to crates.io
#
# Order matters: leaf crates first, then tribev2 (path deps become version deps on upload).
#
# Quick start (from tribev2-rs root):
#
#   ./scripts/publish.sh doctor          # preflight (login, LICENSE, dry-run all)
#   ./scripts/publish.sh dry-run         # package + verify without upload
#   ./scripts/publish.sh publish         # upload to crates.io (asks for confirmation)
#
# Flags:
#
#   --allow-dirty     Pass --allow-dirty to cargo publish
#   --only CRATE      tribev2-audio | tribev2-video | tribev2
#   -y / --yes        Skip confirmation on publish
#
# Examples:
#
#   ./scripts/publish.sh doctor
#   ./scripts/publish.sh dry-run --only tribev2-audio
#   ./scripts/publish.sh publish -y
#   TRIBEV2_PUBLISH_ALLOW_DIRTY=1 ./scripts/publish.sh publish
#
# Prerequisites:
#   - cargo login  (https://crates.io/settings/tokens)
#   - LICENSE at repo root
#   - Workspace path deps include version (see root Cargo.toml)
#
# See also: docs/PUBLISHING.md

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Leaf crates first, then tribev2 (optional deps on audio/video).
CRATES=(tribev2-audio tribev2-video tribev2)

ALLOW_DIRTY=0
SKIP_CONFIRM=0
ONLY_CRATE=""

usage() {
  sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  echo
  echo "--- resolved ---"
  echo "  ROOT=$ROOT"
  echo "  publish order: ${CRATES[*]}"
}

_crates_to_run() {
  if [[ -n "$ONLY_CRATE" ]]; then
    echo "$ONLY_CRATE"
    return
  fi
  printf '%s\n' "${CRATES[@]}"
}

_crate_dir() {
  case "$1" in
    tribev2) echo "$ROOT/crates/tribev2" ;;
    tribev2-audio) echo "$ROOT/crates/tribev2-audio" ;;
    tribev2-video) echo "$ROOT/crates/tribev2-video" ;;
    *)
      echo "publish.sh: unknown crate '$1' (expected: ${CRATES[*]})" >&2
      exit 2
      ;;
  esac
}

_pkg_version() {
  sed -n 's/^version = "\(.*\)"/\1/p' "$(_crate_dir "$1")/Cargo.toml" | head -1
}

_publish_args() {
  local crate=$1 dry=${2:-0}
  local -a args=(-p "$crate")
  [[ "$ALLOW_DIRTY" -eq 1 ]] && args+=(--allow-dirty)
  [[ "$dry" -eq 1 ]] && args+=(--dry-run)
  echo "${args[@]}"
}

_check_git_clean() {
  [[ "$ALLOW_DIRTY" -eq 1 ]] && return 0
  if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    echo "ERROR: working tree not clean. Commit or use --allow-dirty" >&2
    git status --short >&2
    exit 1
  fi
}

_check_cargo_login() {
  if [[ -f "$HOME/.cargo/credentials.toml" || -f "$HOME/.cargo/credentials" ]]; then
    return 0
  fi
  echo "ERROR: not logged in to crates.io — run: cargo login" >&2
  exit 1
}

_check_license() {
  if [[ -f "$ROOT/LICENSE" || -f "$ROOT/LICENSE.txt" || -f "$ROOT/LICENSE-APACHE" ]]; then
    return 0
  fi
  echo "ERROR: missing LICENSE at repo root" >&2
  exit 1
}

_check_workspace_path_versions() {
  grep -q 'tribev2-audio = { version' "$ROOT/Cargo.toml" || {
    echo "ERROR: root Cargo.toml needs version on tribev2-audio workspace dep" >&2
    exit 1
  }
  grep -q 'tribev2-video = { version' "$ROOT/Cargo.toml" || {
    echo "ERROR: root Cargo.toml needs version on tribev2-video workspace dep" >&2
    exit 1
  }
}

_check_crate_metadata() {
  local crate=$1 dir ok=1
  dir="$(_crate_dir "$crate")"
  for key in description license; do
    grep -q "^${key} =" "$dir/Cargo.toml" || {
      echo "  [fail] $crate: missing $key" >&2
      ok=0
    }
  done
  grep -q '^repository =' "$dir/Cargo.toml" || \
    echo "  [warn] $crate: missing repository" >&2
  [[ "$ok" -eq 1 ]]
}

_check_versions_aligned() {
  local va vv vm ws
  va="$(_pkg_version tribev2-audio)"
  vv="$(_pkg_version tribev2-video)"
  vm="$(_pkg_version tribev2)"
  if [[ "$va" != "$vv" || "$va" != "$vm" ]]; then
    echo "WARN: versions differ — audio=$va video=$vv tribev2=$vm" >&2
  fi
  ws="$(sed -n 's/.*tribev2-audio = { version = "\([^"]*\)".*/\1/p' "$ROOT/Cargo.toml")"
  if [[ -n "$ws" && "$ws" != "$va" ]]; then
    echo "ERROR: workspace tribev2-audio version ($ws) != crate ($va)" >&2
    exit 1
  fi
}

_validate_only() {
  if [[ -z "$ONLY_CRATE" ]]; then
    return 0
  fi
  local c
  for c in "${CRATES[@]}"; do
    [[ "$c" == "$ONLY_CRATE" ]] && return 0
  done
  echo "publish.sh: --only $ONLY_CRATE not in: ${CRATES[*]}" >&2
  exit 2
}

cmd_doctor() {
  echo "==> tribev2-rs publish doctor"
  _validate_only
  _check_git_clean
  _check_cargo_login
  _check_license
  _check_workspace_path_versions
  _check_versions_aligned

  local crate
  while IFS= read -r crate; do
    echo "==> $crate v$(_pkg_version "$crate")"
    _check_crate_metadata "$crate"
    echo "  cargo publish $(_publish_args "$crate" 1)"
    cargo publish $(_publish_args "$crate" 1)
  done < <(_crates_to_run)

  echo "doctor: OK"
}

cmd_publish() {
  local dry=$1
  _validate_only
  _check_git_clean
  _check_cargo_login
  _check_license
  _check_workspace_path_versions
  _check_versions_aligned

  local crate
  while IFS= read -r crate; do
    _check_crate_metadata "$crate" || exit 1
  done < <(_crates_to_run)

  if [[ "$dry" -eq 0 && "$SKIP_CONFIRM" -ne 1 ]]; then
    echo ""
    echo "Publish to crates.io:"
    while IFS= read -r crate; do
      echo "  - $crate $(_pkg_version "$crate")"
    done < <(_crates_to_run)
    read -r -p "Continue? [y/N] " ans
    case "$ans" in
      y|Y|yes|YES) ;;
      *) echo "Aborted."; exit 1 ;;
    esac
  fi

  while IFS= read -r crate; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ $crate $(_pkg_version "$crate")"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    # shellcheck disable=SC2046
    cargo publish $( _publish_args "$crate" "$dry" )
  done < <(_crates_to_run)

  echo ""
  if [[ "$dry" -eq 1 ]]; then
    echo "Dry-run OK. Run: ./scripts/publish.sh publish"
  else
    echo "Done:"
    while IFS= read -r crate; do
      echo "  https://crates.io/crates/$crate"
    done < <(_crates_to_run)
  fi
}

main() {
  local action=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --allow-dirty) ALLOW_DIRTY=1; shift ;;
      -y|--yes) SKIP_CONFIRM=1; shift ;;
      --only)
        ONLY_CRATE="${2:?--only requires crate name}"
        shift 2
        ;;
      --only=*) ONLY_CRATE="${1#--only=}"; shift ;;
      -h|--help|help)
        usage
        exit 0
        ;;
      doctor|dry-run|publish)
        action="$1"
        shift
        ;;
      *)
        echo "publish.sh: unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done

  case "${action:-help}" in
    help) usage ;;
    doctor) cmd_doctor ;;
    dry-run) cmd_publish 1 ;;
    publish) cmd_publish 0 ;;
    *)
      echo "publish.sh: unknown command: ${action:-}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
