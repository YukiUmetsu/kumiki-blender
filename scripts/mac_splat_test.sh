#!/usr/bin/env bash
# Mac-first, non-destructive Gaussian-splat smoke test.
set -Eeuo pipefail
usage() {
  cat <<'EOF'
Usage:
  bash scripts/mac_splat_test.sh doctor
  bash scripts/mac_splat_test.sh plan /absolute/path/to/photos [asset-slug]
  bash scripts/mac_splat_test.sh run /absolute/path/to/photos [asset-slug]
  bash scripts/mac_splat_test.sh train-existing /path/to/failed-splat-revision
  bash scripts/mac_splat_test.sh crash-qa /path/to/failed-splat-retry
  bash scripts/mac_splat_test.sh diagnose /path/to/failed-splat-revision
  bash scripts/mac_splat_test.sh cohorts /path/to/failed-splat-revision
  bash scripts/mac_splat_test.sh match-qa /path/to/failed-splat-revision [cohort-source-index.json]
  bash scripts/mac_splat_test.sh pnp-qa /path/to/failed-splat-revision [cohort-source-index.json]
  bash scripts/mac_splat_test.sh photo-advice /path/to/failed-splat-revision [cohort-source-index.json] [front-photo]
  bash scripts/mac_splat_test.sh recover /path/to/failed-splat-revision [model-index]
  bash scripts/mac_splat_test.sh recover-qa /path/to/failed-splat-revision [model-index]
  bash scripts/mac_splat_test.sh mask /path/to/splat-revision [rembg|grabcut]
  bash scripts/mac_splat_test.sh hybrid /path/to/splat-revision /path/to/rembg-mask-revision /path/to/grabcut-mask-revision photo-0002.png photo-0015.png
  bash scripts/mac_splat_test.sh run-masked /path/to/splat-revision /path/to/mask-revision
  bash scripts/mac_splat_test.sh qa /path/to/splat-revision
  bash scripts/mac_splat_test.sh video /path/to/splat-revision

Optional environment:
  OPENSPLAT=/absolute/path/to/opensplat (default: opensplat on PATH)
  SPLAT_TRANSFORM=/absolute/path/to/splat-transform (default: splat-transform)
  OUTPUT_ROOT=/absolute/path/to/revisions (default: .review/mac-splat-smoke)
  MAX_IMAGE_SIZE=2000 ITERATIONS=3000 MAX_VIEWS=12
  DOWNSCALE_FACTOR=4 TRAIN_CPU=0 (train-existing: reuse registered dataset; CPU=1 diagnostic)
  CRASH_QA_TIMEOUT=600 (crash-qa: LLDB timeout in seconds)
  REGISTRATION_PROFILE=standard|high-recall (high-recall: slower, CPU SIFT + guided matching)
  PHOTO_FRONT_REFERENCE=photo-0001.png (optional known frontal view for angle labels)
  PHOTO_VISION_MODEL=<installed-ollama-vision-model> (optional, local image view/framing labels)
  PHOTO_SUBJECT_TYPE=object|statue|building|landscape (default: object)
  VIDEO_SECONDS=3 VIDEO_WIDTH=960 VIDEO_HEIGHT=540
  VIDEO_HERO_PHOTO=photo-0001.png (optional registered view for video)
  MASK_MAX_SIDE=1280 MASK_ITERATIONS=5
Outputs are private, revisioned and never overwritten.
EOF
}
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || die "Missing $1. Install it, then rerun doctor."; }
mode="${1:-}"; [[ -n "$mode" ]] || { usage; exit 2; }
[[ "$mode" == doctor || "$mode" == plan || "$mode" == run || "$mode" == diagnose || "$mode" == cohorts || "$mode" == match-qa || "$mode" == pnp-qa || "$mode" == photo-advice || "$mode" == recover || "$mode" == recover-qa || "$mode" == train-existing || "$mode" == crash-qa || "$mode" == mask || "$mode" == hybrid || "$mode" == run-masked || "$mode" == qa || "$mode" == video ]] || { usage; exit 2; }
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
  python3 -c 'from rembg import new_session; print("OK: rembg (semantic masks)")' 2>/dev/null || printf 'OPTIONAL: rembg (pip install -e ".[mask-ai]")\n'
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
if [[ "$mode" == diagnose ]]; then
  [[ -d "$2" ]] || die "Splat revision directory not found: $2"
  python3 -c 'from PIL import Image' || die 'Install Pillow: pip install Pillow'
  output="$(new_output registration)"
  python3 scripts/diagnose_splat_registration.py "$2" --contact-sheet "$output.jpg" --json-output "$output.json"
  printf '\nRegistration contact sheet: %s.jpg\nRegistration report: %s.json\n' "$output" "$output"
  open "$output.jpg" 2>/dev/null || true
elif [[ "$mode" == cohorts ]]; then
  [[ -d "$2" ]] || die "Splat revision directory not found: $2"
  output="$(new_output cohorts)"
  python3 scripts/prepare_sfm_cohorts.py "$2" --output "$output"
  printf '\nPhoto cohorts and provenance: %s/cohort-plan.json\n' "$output"
elif [[ "$mode" == match-qa ]]; then
  [[ -d "$2" ]] || die "Splat revision directory not found: $2"
  python3 -c 'from PIL import Image' || die 'Install Pillow: pip install Pillow'
  output="$(new_output match-qa)"
  source_args=()
  if [[ $# -ge 3 ]]; then
    [[ -f "$3" ]] || die "Cohort source index not found: $3"
    source_args+=(--source-index "$3")
  fi
  python3 scripts/visualize_sfm_matches.py "$2" --output "$output" "${source_args[@]}"
  open "$output/match-contact-sheet.jpg" 2>/dev/null || true
elif [[ "$mode" == pnp-qa ]]; then
  [[ -d "$2" ]] || die "Splat revision directory not found: $2"
  output="$(new_output pnp-qa)"
  source_args=()
  if [[ $# -ge 3 ]]; then
    [[ -f "$3" ]] || die "Cohort source index not found: $3"
    source_args+=(--source-index "$3")
  fi
  python3 scripts/diagnose_sfm_pnp_support.py "$2" --json-output "$output.json" "${source_args[@]}"
elif [[ "$mode" == photo-advice ]]; then
  [[ -d "$2" ]] || die "Splat revision directory not found: $2"
  output="$(new_output photo-advice)"
  advice_args=()
  if [[ $# -ge 3 ]]; then
    [[ -f "$3" ]] || die "Cohort source index not found: $3"
    advice_args+=(--source-index "$3")
  fi
  if [[ $# -ge 4 ]]; then
    advice_args+=(--front-photo "$4")
  elif [[ -n "${PHOTO_FRONT_REFERENCE:-}" ]]; then
    advice_args+=(--front-photo "$PHOTO_FRONT_REFERENCE")
  fi
  [[ -z "${PHOTO_VISION_MODEL:-}" ]] || advice_args+=(--vision-model "$PHOTO_VISION_MODEL")
  advice_args+=(--subject-type "${PHOTO_SUBJECT_TYPE:-object}")
  python3 scripts/audit_photo_coverage.py "$2" --output "$output" "${advice_args[@]}"
  printf '\nRead: %s/photo-advice.md\nPhoto contact sheet: %s/photo-advice.jpg\n' "$output" "$output"
  [[ -f "$output/photo-advice.jpg" ]] && open "$output/photo-advice.jpg" 2>/dev/null || true
elif [[ "$mode" == recover-qa ]]; then
  [[ -d "$2" ]] || die "Splat revision directory not found: $2"
  output="$(new_output recovery-match-qa)"
  python3 scripts/recover_sfm_tracks.py "$2" --model "${3:-0}" --output "$output" --diagnose-only
elif [[ "$mode" == recover ]]; then
  [[ -d "$2" ]] || die "Splat revision directory not found: $2"
  python3 -c 'import pycolmap' || die 'Install photogrammetry dependencies: pip install -e ".[photogrammetry]"'
  output="$(new_output recovery)"
  python3 scripts/recover_sfm_tracks.py "$2" --model "${3:-0}" --output "$output"
  if python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["status"] == "no-registration-improvement" else 1)' "$output/recovery-report.json"; then
    advice="$(new_output photo-advice)"
    advice_args=(--subject-type "${PHOTO_SUBJECT_TYPE:-object}")
    [[ -z "${PHOTO_FRONT_REFERENCE:-}" ]] || advice_args+=(--front-photo "$PHOTO_FRONT_REFERENCE")
    [[ -z "${PHOTO_VISION_MODEL:-}" ]] || advice_args+=(--vision-model "$PHOTO_VISION_MODEL")
    if python3 scripts/audit_photo_coverage.py "$2" --output "$advice" "${advice_args[@]}"; then
      printf '\nNo registration improvement. Per-photo and missing-angle report: %s/photo-advice.md\n' "$advice"
      [[ -f "$advice/photo-advice.jpg" ]] && open "$advice/photo-advice.jpg" 2>/dev/null || true
    else
      printf 'WARNING: Could not generate photo advice; recovery report remains available.\n' >&2
    fi
  fi
elif [[ "$mode" == crash-qa ]]; then
  [[ -f "$2/retry-report.json" ]] || die "Failed OpenSplat retry report not found: $2/retry-report.json"
  require lldb
  output="$(new_output opensplat-crash-qa)"
  python3 scripts/diagnose_opensplat_crash.py "$2" --output "$output" \
    --timeout "${CRASH_QA_TIMEOUT:-600}"
elif [[ "$mode" == train-existing ]]; then
  [[ -d "$2/dataset" ]] || die "Existing staged dataset not found: $2/dataset"
  check_tool "$trainer"
  output="$(new_output splat-retry)"
  revision="$2"
  # macOS ships Bash 3.2: under set -u, expanding an empty array raises
  # "unbound variable" even when the array was initialized with ().
  # Build positional arguments instead; zero optional arguments are safe.
  set -- "$revision" --output "$output" \
    --opensplat "$trainer" --iterations "${ITERATIONS:-300}" \
    --downscale "${DOWNSCALE_FACTOR:-4}"
  if [[ "${TRAIN_CPU:-0}" == 1 ]]; then
    set -- "$@" --cpu
  fi
  python3 scripts/retry_opensplat_training.py "$@"
elif [[ "$mode" == plan || "$mode" == run ]]; then
  photos="$2"; [[ -d "$photos" ]] || die "Photo directory not found: $photos"
  slug="${3:-mac-splat-test}"
  [[ "$slug" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || die 'Asset slug must contain only letters, numbers, hyphens and underscores.'
  count="$(find "$photos" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l | tr -d ' ')"
  (( count >= 8 && count <= 256 )) || die "Need 8–256 photos; found $count."
  profile="${REGISTRATION_PROFILE:-standard}"
  [[ "$profile" == standard || "$profile" == high-recall ]] || die "REGISTRATION_PROFILE must be standard or high-recall"
  max_image_size="${MAX_IMAGE_SIZE:-$( [[ "$profile" == high-recall ]] && echo 3200 || echo 2000 )}"
  output="$(new_output splat)"
  printf 'Photos: %s | Registration: %s | Max image size: %s | Output: %s\n' "$count" "$profile" "$max_image_size" "$output"
  kumiki-blender plan-splats --photos "$photos" --asset-id "$slug" --output "$output" --opensplat "$trainer" \
    --registration-profile "$profile"
  [[ "$mode" == plan ]] && exit 0
  check_tool "$trainer"
  python3 -c 'import pycolmap' || die 'Install photogrammetry dependencies: pip install -e ".[photogrammetry]"'
  if kumiki-blender build-splats --photos "$photos" --asset-id "$slug" --output "$output" \
    --engine pycolmap --opensplat "$trainer" --max-image-size "$max_image_size" \
    --registration-profile "$profile" --iterations "${ITERATIONS:-3000}"; then
    printf '\nTraining output: %s\nRun QA: bash scripts/mac_splat_test.sh qa "%s"\n' "$output" "$output"
  else
    status=$?
    failed_stage=""
    if [[ -f "$output/splat-report.json" ]]; then
      failed_stage="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("failed_stage") or "")' "$output/splat-report.json" 2>/dev/null || true)"
    fi
    if [[ "$failed_stage" == opensplat* ]]; then
      printf '\nCamera registration succeeded; OpenSplat training failed (%s).\n' "$failed_stage" >&2
      printf 'Native training log: %s/logs/opensplat.log\n' "$output" >&2
      if [[ -f "$output/logs/opensplat.log" ]]; then
        printf '\nLast 35 trainer log lines:\n' >&2
        tail -n 35 "$output/logs/opensplat.log" >&2
      fi
      printf '\nRetry training without repeating SfM:\n  DOWNSCALE_FACTOR=4 ITERATIONS=300 bash scripts/mac_splat_test.sh train-existing "%s"\n' "$output" >&2
      printf 'If MPS still crashes, set TRAIN_CPU=1 to isolate the native GPU path.\n' >&2
    elif [[ -f "$output/sparse-reconstruction/database.db" ]]; then
      diagnostic="$(new_output registration)"
      printf '\nCamera registration incomplete. Generating read-only registration diagnosis...\n' >&2
      if python3 scripts/diagnose_splat_registration.py "$output" --contact-sheet "$diagnostic.jpg" --json-output "$diagnostic.json"; then
        printf '\nRegistration contact sheet: %s.jpg\nRegistration report: %s.json\n' "$diagnostic" "$diagnostic" >&2
        open "$diagnostic.jpg" 2>/dev/null || true
      else
        printf 'WARNING: Registration diagnosis failed; original build exit code preserved.\n' >&2
      fi
      if [[ -d "$output/sparse-reconstruction/text-models" ]]; then
        advice="$(new_output photo-advice)"
        advice_args=()
        [[ -z "${PHOTO_FRONT_REFERENCE:-}" ]] || advice_args+=(--front-photo "$PHOTO_FRONT_REFERENCE")
        [[ -z "${PHOTO_VISION_MODEL:-}" ]] || advice_args+=(--vision-model "$PHOTO_VISION_MODEL")
        advice_args+=(--subject-type "${PHOTO_SUBJECT_TYPE:-object}")
        if python3 scripts/audit_photo_coverage.py "$output" --output "$advice" "${advice_args[@]}"; then
          printf '\nPhoto value + missing-angle recommendations: %s/photo-advice.md\n' "$advice" >&2
        else
          printf 'WARNING: Photo advice unavailable; original build exit code preserved.\n' >&2
        fi
      fi
    fi
    printf 'Failed splat revision retained at: %s\n' "$output" >&2
    exit "$status"
  fi
elif [[ "$mode" == mask || "$mode" == hybrid || "$mode" == run-masked ]]; then
  revision="$2"
  [[ -f "$revision/splat-report.json" && -d "$revision/sparse-reconstruction" ]] || die "Expected a completed multiview splat revision with sparse-reconstruction: $revision"
  reconstruction="$revision/sparse-reconstruction"
  if [[ "$mode" == mask ]]; then
    segmenter="${3:-rembg}"
    [[ "$segmenter" == rembg || "$segmenter" == grabcut ]] || die "Segmenter must be rembg or grabcut"
    if [[ "$segmenter" == rembg ]]; then
      python3 -c 'from rembg import new_session' || die 'Install semantic mask support: pip install -e ".[mask-ai]"'
    else
      python3 -c 'import cv2' || die 'Install GrabCut support: pip install -e ".[masks]"'
    fi
    output="$(new_output mask)"
    kumiki-blender prepare-dense-masks --reconstruction "$reconstruction" --output "$output" \
      --segmenter "$segmenter" --max-side "${MASK_MAX_SIDE:-1280}" --iterations "${MASK_ITERATIONS:-5}"
    printf '\nInspect EVERY subject mask before training: %s/mask-contact-sheet.jpg\n' "$output"
    printf 'To train with reviewed masks: bash scripts/mac_splat_test.sh run-masked "%s" "%s"\n' "$revision" "$output"
    open "$output/mask-contact-sheet.jpg" 2>/dev/null || true
  elif [[ "$mode" == hybrid ]]; then
    [[ $# -ge 5 ]] || die "hybrid requires a splat revision, two mask revisions and at least one photo filename"
    primary_masks="$3"
    alternate_masks="$4"
    shift 4
    output="$(new_output mask)"
    kumiki-blender compose-dense-masks --reconstruction "$reconstruction" \
      --primary-mask-revision "$primary_masks" --alternate-mask-revision "$alternate_masks" \
      --output "$output" --use-alternate "$@"
    printf '\nReview every hybrid mask: %s/mask-contact-sheet.jpg\n' "$output"
    printf 'Optional leakage cleanup: kumiki-blender refine-dense-masks --reconstruction "%s" --mask-revision "%s" --output /path/to/new-mask-revision --edits /path/to/edits.json\n' "$reconstruction" "$output"
    printf 'Retrain after review: bash scripts/mac_splat_test.sh run-masked "%s" "%s"\n' "$revision" "$output"
    open "$output/mask-contact-sheet.jpg" 2>/dev/null || true
  else
    [[ $# -eq 3 ]] || die "run-masked requires splat and mask revision paths"
    masks="$3"
    [[ -f "$masks/mask-report.json" && -d "$masks/masks" ]] || die "Expected prepare-dense-masks revision: $masks"
    # Match the mask report's immutable reconstruction source, so a mask
    # set cannot be accidentally applied to differently registered photos.
    python3 -c 'import json,pathlib,sys; report=json.loads((pathlib.Path(sys.argv[1])/"mask-report.json").read_text()); source=pathlib.Path(report["source_revision"]).resolve(); expected=pathlib.Path(sys.argv[2]).resolve(); sys.exit(0 if source == expected and report["status"].startswith("candidate") else 3)' "$masks" "$reconstruction" || die "Mask provenance check failed"
    check_tool "$trainer"
    output="$(new_output splat)"
    kumiki-blender build-splats --reconstruction "$reconstruction" --rectified-masks "$masks/masks" \
      --output "$output" --opensplat "$trainer" --iterations "${ITERATIONS:-3000}"
    printf '\nMasked training output: %s\nRun QA: bash scripts/mac_splat_test.sh qa "%s"\n' "$output" "$output"
  fi
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
  video_args=()
  if [[ -n "${VIDEO_HERO_PHOTO:-}" ]]; then video_args+=(--hero-photo "$VIDEO_HERO_PHOTO"); fi
  kumiki-blender plan-splat-video --splat "$2" --output "$output" "${video_args[@]}"
  kumiki-blender render-splat-video --splat "$2" --output "$output" --splat-transform "$rasterizer" \
    --seconds "${VIDEO_SECONDS:-3}" --width "${VIDEO_WIDTH:-960}" --height "${VIDEO_HEIGHT:-540}" \
    "${video_args[@]}" --viewer
  printf '\nVideo: %s/video.mp4\n' "$output"
  open "$output/video.mp4" 2>/dev/null || true
fi
