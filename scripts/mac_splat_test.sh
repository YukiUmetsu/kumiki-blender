#!/usr/bin/env bash
# Mac-first, non-destructive Gaussian-splat smoke test.
set -Eeuo pipefail
usage() {
  cat <<'EOF'
Usage:
  bash scripts/mac_splat_test.sh doctor
  bash scripts/mac_splat_test.sh plan /absolute/path/to/photos [asset-slug]
  bash scripts/mac_splat_test.sh run /absolute/path/to/photos [asset-slug]
  bash scripts/mac_splat_test.sh qa /path/to/splat-revision
  bash scripts/mac_splat_test.sh video /path/to/splat-revision

Optional environment:
  OPENSPLAT=/absolute/path/to/opensplat (default: opensplat on PATH)
  SPLAT_TRANSFORM=/absolute/path/to/splat-transform (default: splat-transform)
  OUTPUT_ROOT=/absolute/path/to/revisions (default: .review/mac-splat-smoke)
  MAX_IMAGE_SIZE=2000 ITERATIONS=3000 MAX_VIEWS=12
  VIDEO_SECONDS=3 VIDEO_WIDTH=960 VIDEO_HEIGHT=540
Outputs are private, revisioned and never overwritten.
EOF
}
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || die "Missing $1. Install it, then rerun doctor."; }
mode="${1:-}"; [[ -n "$mode" ]] || { usage; exit 2; }
[[ "$mode" == doctor || "$mode" == plan || "$mode" == run || "$mode" == qa || "$mode" == video ]] || { usage; exit 2; }
if [[ "$mode" != doctor ]]; then [[ $# -ge 2 ]] || { usage; exit 2; }; fi
[[ "$(uname -s)" == Darwin ]] || printf 'WARNING: Intended for macOS; current OS is %s\n' "$(uname -s)" >&2
root="${OUTPUT_ROOT:-.review/mac-splat-smoke}"
trainer="${OPENSPLAT:-opensplat}"
rasterizer="${SPLAT_TRANSFORM:-splat-transform}"
check_tool() { if [[ "$1" == */* ]]; then [[ -x "$1" ]] || die "Not executable: $1"; else require "$1"; fi; }
if [[ "$mode" == doctor ]]; then
  printf 'Platform: %s / %s\n' "$(uname -s)" "$(uname -m)"
  for tool in python3 kumiki-blender "$trainer" "$rasterizer"; do
    if [[ "$tool" == */* ]]; then [[ -x "$tool" ]] && printf 'OK: %s\n' "$tool" || printf 'MISSING: %s\n' "$tool"
    elif command -v "$tool" >/dev/null 2>&1; then printf 'OK: %s (%s)\n' "$tool" "$(command -v "$tool")"
    else printf 'MISSING: %s\n' "$tool"; fi
  done
  python3 -c 'import pycolmap; print("OK: pycolmap")' 2>/dev/null || printf 'MISSING: pycolmap (pip install -e ".[photogrammetry]")\n'
  python3 -c 'from PIL import Image; print("OK: Pillow")' 2>/dev/null || printf 'MISSING: Pillow (pip install Pillow)\n'
  if command -v "$rasterizer" >/dev/null 2>&1 || [[ -x "$rasterizer" ]]; then "$rasterizer" --list-gpus || printf 'WARNING: GPU discovery failed\n'; fi
  printf 'OpenSplat must be built with Metal/MPS; a detected executable alone does not prove GPU acceleration.\n'
  exit 0
fi
require kumiki-blender
new_output() {
  mkdir -p "$root"
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local candidate="$root/$1-$stamp-$$"
  [[ ! -e "$candidate" ]] || die "Output already exists: $candidate"
  printf '%s\n' "$candidate"
}
if [[ "$mode" == plan || "$mode" == run ]]; then
  photos="$2"; [[ -d "$photos" ]] || die "Photo directory not found: $photos"
  slug="${3:-mac-splat-test}"
  [[ "$slug" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || die 'Asset slug must contain only letters, numbers, hyphens and underscores.'
  count="$(find "$photos" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l | tr -d ' ')"
  (( count >= 8 && count <= 256 )) || die "Need 8–256 photos; found $count."
  output="$(new_output splat)"
  printf 'Photos: %s | Output: %s\n' "$count" "$output"
  kumiki-blender plan-splats --photos "$photos" --asset-id "$slug" --output "$output"
  [[ "$mode" == plan ]] && exit 0
  check_tool "$trainer"
  python3 -c 'import pycolmap' || die 'Install photogrammetry dependencies: pip install -e ".[photogrammetry]"'
  kumiki-blender build-splats --photos "$photos" --asset-id "$slug" --output "$output" \
    --engine pycolmap --opensplat "$trainer" --max-image-size "${MAX_IMAGE_SIZE:-2000}" --iterations "${ITERATIONS:-3000}"
  printf '\nTraining output: %s\nRun QA: bash scripts/mac_splat_test.sh qa "%s"\n' "$output" "$output"
elif [[ "$mode" == qa ]]; then
  [[ -d "$2" ]] || die "Splat revision directory not found: $2"
  check_tool "$rasterizer"
  python3 -c 'from PIL import Image' || die 'Install Pillow: pip install Pillow'
  output="$(new_output qa)"
  kumiki-blender plan-splat-view-qa --splat "$2" --output "$output" --max-views "${MAX_VIEWS:-12}"
  kumiki-blender compare-splat-views --splat "$2" --output "$output" --max-views "${MAX_VIEWS:-12}" \
    --max-dimension 1280 --splat-transform "$rasterizer"
  printf '\nOpen: %s/multi-angle-contact-sheet.jpg\nMetrics: %s/visual-qa-report.json\n' "$output" "$output"
  open "$output/multi-angle-contact-sheet.jpg" 2>/dev/null || true
elif [[ "$mode" == video ]]; then
  [[ -e "$2" ]] || die "Splat not found: $2"
  check_tool "$rasterizer"
  require ffmpeg; require ffprobe
  output="$(new_output video)"
  kumiki-blender plan-splat-video --splat "$2" --output "$output"
  kumiki-blender render-splat-video --splat "$2" --output "$output" --splat-transform "$rasterizer" \
    --seconds "${VIDEO_SECONDS:-3}" --width "${VIDEO_WIDTH:-960}" --height "${VIDEO_HEIGHT:-540}" --viewer
  printf '\nVideo: %s/video.mp4\n' "$output"
  open "$output/video.mp4" 2>/dev/null || true
fi
