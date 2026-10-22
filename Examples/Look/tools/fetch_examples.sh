#!/bin/bash
# fetch_examples.sh - puts the three example photos of Examples/Look into Examples/Look/Examples/, where the app
# bundles them from, the example video of the Watch screen into Examples/Look/Videos/, and the Watch screen's effect
# sounds into Examples/Look/Sounds/, where the app bundles them from.
#
# The repository holds no images, videos or sounds, so the examples are kept by reference: each entry of
# tools/examples.json (next to this script) names the URL its bytes come from, their size and their sha256. This script
# downloads each photo with curl, checks both numbers, and writes it under the file name the app reads. The video is not
# the URL's bytes: its entry ("kind": "video") names the original, which the script downloads into Videos/src/ and
# checks the same way, and the cut the app plays (start, length, size, frame rate, CRF), which the script makes with
# ffmpeg into Videos/<file> and checks (H.264, its size, its frame rate, its length, no audio; with the ffmpeg the entry
# names, the same bytes as cut_sha256). A sound ("kind": "sound") is the same: the original goes into SoundSources/<id>/
# (not into Sounds/, which the app bundles whole), the cut (mono, 48 kHz, start, length, fades, gain, AAC) into
# Sounds/<file>, checked for its form, its length and its loudness. A file already in place with the right size and
# sha256 (a clip: that passes its checks) is left alone.
#
#   tools/fetch_examples.sh                  fetch what is missing, then check everything
#   tools/fetch_examples.sh --check          only check the files already there (no network, no encode)
#   tools/fetch_examples.sh --no-video       without the video;  --no-sounds  without the sounds
#   tools/fetch_examples.sh [--check] [--no-video] [--no-sounds] [--videos VDIR] [--sounds SDIR]
#                           [--sound-sources SSDIR] [DIR]
#                                            DIR for the photos, VDIR for the video (default Examples/Look/Videos),
#                                            SDIR for the sounds (default Examples/Look/Sounds), SSDIR for their
#                                            originals (default Examples/Look/SoundSources)
#
# The photos need bash, curl and shasum only (all three ship with macOS). The video and the sounds also need ffmpeg and
# ffprobe (brew install ffmpeg); without them the script says so, skips them and still checks the photos — and the app
# plays the system chime for an effect without its sound. Exit status 0 when everything it handled is in place and
# matches, 1 otherwise (the reason per file is printed), 2 on a usage error or an unreadable examples.json. The photos,
# the video and the sounds are CC0 1.0 (examples.json, README: Example photos, Example video, Reaction sounds). When a
# URL stops serving these bytes, the script says so and names where the file can be found again.

set -u

here=$(cd "$(dirname "$0")" && pwd)
manifest=$here/examples.json
user_agent="coreai-kit-look-examples/1 (+https://github.com/john-rocky/coreai-kit)"

check_only=0
with_video=1
with_sounds=1
dest=""
vdest=""
sdest=""
ssdest=""
while [ $# -gt 0 ]; do
  case $1 in
    --check) check_only=1 ;;
    --no-video) with_video=0 ;;
    --no-sounds) with_sounds=0 ;;
    --videos) shift; vdest=${1:-}; [ -n "$vdest" ] || { echo "--videos needs a directory (see --help)" >&2; exit 2; } ;;
    --sounds) shift; sdest=${1:-}; [ -n "$sdest" ] || { echo "--sounds needs a directory (see --help)" >&2; exit 2; } ;;
    --sound-sources) shift; ssdest=${1:-}; [ -n "$ssdest" ] || { echo "--sound-sources needs a directory (see --help)" >&2; exit 2; } ;;
    -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    *)
      if [ -n "$dest" ]; then echo "one destination only (see --help)" >&2; exit 2; fi
      dest=$1 ;;
  esac
  shift
done
[ -n "$dest" ] || dest=$(cd "$here/.." && pwd)/Examples
[ -n "$vdest" ] || vdest=$(cd "$here/.." && pwd)/Videos
[ -n "$sdest" ] || sdest=$(cd "$here/.." && pwd)/Sounds
[ -n "$ssdest" ] || ssdest=$(cd "$here/.." && pwd)/SoundSources

if [ ! -r "$manifest" ]; then
  echo "cannot read $manifest" >&2
  exit 2
fi

# One line per entry, read from examples.json as it is written (one "key": value pair per line), fields separated by
# the unit separator \037 (a tab would let an empty field shift the rest): id, kind, file, url, bytes, sha256,
# openverse_record, source_page, source_file, cut_start, cut_seconds, cut_fps, cut_crf, cut_width, cut_height,
# cut_sha256, cut_ffmpeg, fade_in, fade_out, gain_db, loudness_lufs (from source_file on: a video's or a sound's;
# empty for a photo).
entries() {
  awk '
    /^ *"(id|kind|file|url|bytes|sha256|openverse_record|source_page|source_file|cut_start|cut_seconds|cut_fps|cut_crf|cut_width|cut_height|cut_sha256|cut_ffmpeg|fade_in|fade_out|gain_db|loudness_lufs)": / {
      line = $0
      sub(/^ *"/, "", line)
      key = line; sub(/".*/, "", key)
      val = line; sub(/^[^:]*: */, "", val); sub(/,? *$/, "", val); gsub(/^"|"$/, "", val)
      rec[key] = val
      next
    }
    /^ *},? *$/ {
      if ("id" in rec)
        print rec["id"] "\037" rec["kind"] "\037" rec["file"] "\037" rec["url"] "\037" rec["bytes"] "\037" rec["sha256"] "\037" \
          rec["openverse_record"] "\037" rec["source_page"] "\037" rec["source_file"] "\037" rec["cut_start"] "\037" \
          rec["cut_seconds"] "\037" rec["cut_fps"] "\037" rec["cut_crf"] "\037" rec["cut_width"] "\037" rec["cut_height"] "\037" \
          rec["cut_sha256"] "\037" rec["cut_ffmpeg"] "\037" rec["fade_in"] "\037" rec["fade_out"] "\037" rec["gain_db"] "\037" \
          rec["loudness_lufs"]
      split("", rec)
    }
  ' "$manifest"
}

size_of() { wc -c < "$1" | tr -d ' '; }
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }

# Downloads $1 into a temporary file in directory $2 with curl (max $3 s), checks it against $4 bytes and sha256 $5,
# and moves it to $6. Prints one line; returns 0 when the file is in place.
download() {
  local url=$1 dir=$2 max=$3 bytes=$4 sha=$5 path=$6 tmp rc got_bytes got_sha
  if ! mkdir -p "$dir"; then echo "cannot create $dir"; return 1; fi
  tmp=$(mktemp "$dir/.fetch-XXXXXX") || { echo "cannot write in $dir"; return 1; }
  curl -fsSL --retry 2 --connect-timeout 20 --max-time "$max" -A "$user_agent" -o "$tmp" "$url" < /dev/null
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "could not download $url (curl exit $rc). Nothing was written."
    rm -f "$tmp"; return 1
  fi
  got_bytes=$(size_of "$tmp")
  got_sha=$(sha_of "$tmp")
  if [ "$got_bytes" != "$bytes" ] || [ "$got_sha" != "$sha" ]; then
    echo "$url sent $got_bytes bytes, sha256 $got_sha; expected $bytes bytes, sha256 $sha. Nothing was written."
    rm -f "$tmp"; return 1
  fi
  chmod 644 "$tmp"
  mv -f "$tmp" "$path"
  echo "$got_bytes bytes, sha256 ${sha:0:12}... from $url"
}

# --- the photos
total=0
failed=0
while IFS=$'\037' read -r id kind file url bytes sha record page _; do
  case $kind in video|sound) continue ;; esac
  total=$((total + 1))
  if [ -z "$file" ] || [ -z "$url" ] || [ -z "$bytes" ] || [ ${#sha} -ne 64 ]; then
    echo "FAILED  ${id:-?}: examples.json lacks file, url, bytes or sha256 for this photo"
    failed=$((failed + 1))
    continue
  fi
  path=$dest/$file
  if [ -f "$path" ]; then
    have_bytes=$(size_of "$path")
    have_sha=$(sha_of "$path")
    if [ "$have_bytes" = "$bytes" ] && [ "$have_sha" = "$sha" ]; then
      echo "ok      $id: $path ($bytes bytes, sha256 ${sha:0:12}...) already in place"
      continue
    fi
    if [ $check_only = 1 ]; then
      echo "DIFFERS $id: $path has $have_bytes bytes, sha256 $have_sha; expected $bytes bytes, sha256 $sha" \
        "(run without --check to fetch it again)"
      failed=$((failed + 1))
      continue
    fi
  elif [ $check_only = 1 ]; then
    echo "MISSING $id: $path (run without --check to fetch it)"
    failed=$((failed + 1))
    continue
  fi
  if why=$(download "$url" "$dest" 120 "$bytes" "$sha" "$path"); then
    echo "fetched $id: $path ($why)"
  else
    echo "FAILED  $id: $why The source no longer serves the same file: find the photo again from $record or $page."
    failed=$((failed + 1))
  fi
done < <(entries)

if [ $total -eq 0 ]; then
  echo "no photos found in $manifest" >&2
  exit 2
fi
if [ $failed -ne 0 ]; then
  echo "$failed of $total example photos are not in place."
else
  echo "all $total example photos are in place."
fi

# --- the video
# The clip's checks: H.264, yuv420p, its width and height, the frame rate, no audio, the length within a frame of
# cut_seconds. Prints nothing and returns 0 when all hold, else prints the first difference.
clip_differs() {
  local clip=$1 seconds=$2 fps=$3 cw=$4 ch=$5 v a d
  v=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,pix_fmt,r_frame_rate \
    -of default=nw=1:nk=1 "$clip" 2>/dev/null | paste -sd' ' -)
  a=$(ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$clip" 2>/dev/null)
  if [ "$v" != "h264 $cw $ch yuv420p $fps/1" ]; then echo "is [$v], expected [h264 $cw $ch yuv420p $fps/1]"; return 1; fi
  if [ -n "$a" ]; then echo "has an audio stream"; return 1; fi
  d=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$clip" 2>/dev/null)
  if ! awk -v d="$d" -v s="$seconds" -v f="$fps" 'BEGIN { x = d - s; if (x < 0) x = -x; exit !(x <= 1.0 / f + 0.001) }'; then
    echo "lasts $d s, expected $seconds s"; return 1
  fi
  return 0
}

vtotal=0
vfailed=0
if [ $with_video = 1 ]; then
  if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "skipped the example video: the app plays a cut of its original, made with ffmpeg, and ffmpeg / ffprobe are not" \
      "installed (brew install ffmpeg, then run this script again). The photos do not need it."
  else
    ffv=$(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')
    while IFS=$'\037' read -r id kind file url bytes sha record page source_file start seconds fps crf cw ch cut_sha cut_ffmpeg _; do
      [ "$kind" = video ] || continue
      vtotal=$((vtotal + 1))
      if [ -z "$file" ] || [ -z "$url" ] || [ -z "$bytes" ] || [ ${#sha} -ne 64 ] || [ -z "$source_file" ] || [ -z "$start" ] \
         || [ -z "$seconds" ] || [ -z "$fps" ] || [ -z "$crf" ] || [ -z "$cw" ] || [ -z "$ch" ]; then
        echo "FAILED  ${id:-?}: examples.json lacks file, url, bytes, sha256, source_file or a cut_ value for this video"
        vfailed=$((vfailed + 1))
        continue
      fi
      clip=$vdest/$file
      src=$vdest/src/$source_file
      same=0
      [ "$ffv" = "$cut_ffmpeg" ] && [ ${#cut_sha} -eq 64 ] && same=1
      if [ -f "$clip" ]; then
        why=$(clip_differs "$clip" "$seconds" "$fps" "$cw" "$ch")
        if [ -z "$why" ] && [ $same = 1 ] && [ "$(sha_of "$clip")" != "$cut_sha" ]; then
          why="has sha256 $(sha_of "$clip"), expected $cut_sha (ffmpeg $cut_ffmpeg)"
        fi
        if [ -z "$why" ]; then
          if [ $same = 1 ]; then
            echo "ok      $id: $clip (${seconds} s from ${start} s of the original, sha256 ${cut_sha:0:12}..., the same bytes as ffmpeg $cut_ffmpeg makes)"
          else
            echo "ok      $id: $clip (${seconds} s from ${start} s of the original, ${cw}x${ch} H.264 ${fps} fps, no audio; bytes not compared: ffmpeg $ffv, the entry's is $cut_ffmpeg)"
          fi
          continue
        fi
        if [ $check_only = 1 ]; then
          echo "DIFFERS $id: $clip $why (run without --check to make it again)"
          vfailed=$((vfailed + 1)); continue
        fi
      elif [ $check_only = 1 ]; then
        echo "MISSING $id: $clip (run without --check to fetch the original and cut it)"
        vfailed=$((vfailed + 1)); continue
      fi
      # the original: in place, or downloaded (283 MB for the flyover: up to 30 min on a slow line)
      if [ -f "$src" ] && [ "$(size_of "$src")" = "$bytes" ] && [ "$(sha_of "$src")" = "$sha" ]; then
        echo "ok      $id: the original $src ($bytes bytes, sha256 ${sha:0:12}...) already in place"
      elif why=$(download "$url" "$vdest/src" 1800 "$bytes" "$sha" "$src"); then
        echo "fetched $id: the original $src ($why)"
      else
        echo "FAILED  $id: $why The source no longer serves the same file: see $page."
        vfailed=$((vfailed + 1)); continue
      fi
      # the cut: -ss before -i (frame exact from the keyframe before), scaled, 30 fps, BT.709 tags, fixed encoder
      # threads (x264's output depends on its thread count), no metadata: the same ffmpeg writes the same bytes
      tmpclip=$(mktemp "$vdest/.cut-XXXXXX") || { echo "FAILED  $id: cannot write in $vdest"; vfailed=$((vfailed + 1)); continue; }
      ffmpeg -nostdin -loglevel error -y -ss "$start" -i "$src" -t "$seconds" -map 0:v:0 -an -sn -dn \
        -vf "scale=$cw:$ch,fps=$fps,format=yuv420p" -c:v libx264 -preset medium -crf "$crf" -threads 8 \
        -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
        -movflags +faststart -map_metadata -1 -map_chapters -1 -fflags +bitexact -f mp4 "$tmpclip"
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "FAILED  $id: ffmpeg exit $rc while cutting $src. Nothing was written."
        rm -f "$tmpclip"; vfailed=$((vfailed + 1)); continue
      fi
      why=$(clip_differs "$tmpclip" "$seconds" "$fps" "$cw" "$ch")
      if [ -n "$why" ]; then
        echo "FAILED  $id: the new clip $why. Nothing was written."
        rm -f "$tmpclip"; vfailed=$((vfailed + 1)); continue
      fi
      chmod 644 "$tmpclip"
      mv -f "$tmpclip" "$clip"
      got=$(sha_of "$clip")
      if [ $same = 1 ] && [ "$got" != "$cut_sha" ]; then
        echo "cut     $id: $clip (sha256 $got, not the entry's $cut_sha although ffmpeg is $cut_ffmpeg: its form is checked, the bytes differ)"
      else
        echo "cut     $id: $clip (${seconds} s from ${start} s of $source_file, ${cw}x${ch}, sha256 $got, ffmpeg $ffv)"
      fi
    done < <(entries)
    if [ $vfailed -ne 0 ]; then
      echo "$vfailed of $vtotal example videos are not in place."
    elif [ $vtotal -gt 0 ]; then
      echo "the example video is in place."
    fi
  fi
fi

# --- the sounds
# The filter chain of a sound's cut: mono and 48 kHz first, then the trim (sample exact), the fades and the gain.
sound_chain() {
  local start=$1 seconds=$2 fin=$3 fout=$4 gain=$5 fout_at
  fout_at=$(awk -v d="$seconds" -v f="$fout" 'BEGIN { printf "%.3f", d - f }')
  printf 'aformat=sample_fmts=fltp:channel_layouts=mono,aresample=48000,atrim=start=%s:duration=%s,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=%s,afade=t=out:st=%s:d=%s,volume=%sdB' \
    "$start" "$seconds" "$fin" "$fout_at" "$fout" "$gain"
}

# A sound clip's checks: AAC, 48 kHz, mono, no video, cut_seconds within 0.03 s and at most 3 s, the loudness within
# 1 LU of loudness_lufs and the true peak below 0 dBTP (ffmpeg's ebur128). Prints nothing and returns 0 when all hold,
# else prints the first difference.
sound_differs() {
  local clip=$1 seconds=$2 target=$3 s d lt
  s=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate,channels -of default=nw=1:nk=1 \
    "$clip" 2>/dev/null | paste -sd' ' -)
  if [ "$s" != "aac 48000 1" ]; then echo "is [$s], expected [aac 48000 1]"; return 1; fi
  if [ -n "$(ffprobe -v error -select_streams v -show_entries stream=index -of csv=p=0 "$clip" 2>/dev/null)" ]; then
    echo "has a video stream"; return 1
  fi
  d=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$clip" 2>/dev/null)
  if ! awk -v d="$d" -v s="$seconds" 'BEGIN { x = d - s; if (x < 0) x = -x; exit !(x <= 0.03 && d <= 3.0) }'; then
    echo "lasts $d s, expected $seconds s (at most 3 s)"; return 1
  fi
  lt=$(ffmpeg -nostdin -hide_banner -nostats -loglevel info -i "$clip" -map 0:a:0 -af "ebur128=peak=true:framelog=quiet" \
    -f null - 2>&1 | awk '
      /Integrated loudness:/ { sect = "I" } /True peak:/ { sect = "TP" }
      sect == "I" && /^ *I:/ { i = $2 }
      sect == "TP" && /^ *Peak:/ { tp = $2 }
      END { if (i == "" || tp == "") exit 1; print i, tp }') || { echo "its loudness could not be measured"; return 1; }
  if ! awk -v i="${lt% *}" -v tp="${lt#* }" -v t="$target" 'BEGIN { x = i - t; if (x < 0) x = -x; exit !(x <= 1.0 && tp < 0) }'; then
    echo "is ${lt% *} LUFS with a true peak of ${lt#* } dBTP, expected $target LUFS within 1 LU and a peak below 0"; return 1
  fi
  return 0
}

stotal=0
sfailed=0
if [ $with_sounds = 1 ]; then
  if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "skipped the effect sounds: the app plays cuts of their originals, made with ffmpeg, and ffmpeg / ffprobe are not" \
      "installed (brew install ffmpeg, then run this script again). Without them an effect plays the system chime."
  else
    ffv=$(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')
    while IFS=$'\037' read -r id kind file url bytes sha record page source_file start seconds fps crf cw ch cut_sha cut_ffmpeg \
      fin fout gain target _; do
      [ "$kind" = sound ] || continue
      stotal=$((stotal + 1))
      if [ -z "$file" ] || [ -z "$url" ] || [ -z "$bytes" ] || [ ${#sha} -ne 64 ] || [ -z "$source_file" ] || [ -z "$start" ] \
         || [ -z "$seconds" ] || [ -z "$fin" ] || [ -z "$fout" ] || [ -z "$gain" ] || [ -z "$target" ]; then
        echo "FAILED  ${id:-?}: examples.json lacks file, url, bytes, sha256, source_file, a cut_ / fade_ value, gain_db or loudness_lufs for this sound"
        sfailed=$((sfailed + 1))
        continue
      fi
      clip=$sdest/$file
      src=$ssdest/$id/$source_file
      same=0
      [ "$ffv" = "$cut_ffmpeg" ] && [ ${#cut_sha} -eq 64 ] && same=1
      if [ -f "$clip" ]; then
        why=$(sound_differs "$clip" "$seconds" "$target")
        if [ -z "$why" ] && [ $same = 1 ] && [ "$(sha_of "$clip")" != "$cut_sha" ]; then
          why="has sha256 $(sha_of "$clip"), expected $cut_sha (ffmpeg $cut_ffmpeg)"
        fi
        if [ -z "$why" ]; then
          if [ $same = 1 ]; then
            echo "ok      $id: $clip (${seconds} s from ${start} s of the original, sha256 ${cut_sha:0:12}..., the same bytes as ffmpeg $cut_ffmpeg makes)"
          else
            echo "ok      $id: $clip (${seconds} s from ${start} s of the original, AAC 48 kHz mono, $target LUFS within 1 LU; bytes not compared: ffmpeg $ffv, the entry's is $cut_ffmpeg)"
          fi
          continue
        fi
        if [ $check_only = 1 ]; then
          echo "DIFFERS $id: $clip $why (run without --check to make it again)"
          sfailed=$((sfailed + 1)); continue
        fi
      elif [ $check_only = 1 ]; then
        echo "MISSING $id: $clip (run without --check to fetch the original and cut it)"
        sfailed=$((sfailed + 1)); continue
      fi
      # the original: in place, or downloaded (0.3 to 2.6 MB)
      if [ -f "$src" ] && [ "$(size_of "$src")" = "$bytes" ] && [ "$(sha_of "$src")" = "$sha" ]; then
        echo "ok      $id: the original $src ($bytes bytes, sha256 ${sha:0:12}...) already in place"
      elif why=$(download "$url" "$ssdest/$id" 120 "$bytes" "$sha" "$src"); then
        echo "fetched $id: the original $src ($why)"
      else
        echo "FAILED  $id: $why The source no longer serves the same file: see $page."
        sfailed=$((sfailed + 1)); continue
      fi
      # the cut: bit-exact flags keep the encoder's version string out of the file, so the same ffmpeg writes the same bytes
      mkdir -p "$sdest" || { echo "FAILED  $id: cannot create $sdest"; sfailed=$((sfailed + 1)); continue; }
      tmpclip=$(mktemp "$sdest/.cut-XXXXXX") || { echo "FAILED  $id: cannot write in $sdest"; sfailed=$((sfailed + 1)); continue; }
      ffmpeg -nostdin -loglevel error -y -i "$src" -map 0:a:0 -vn -sn -dn \
        -af "$(sound_chain "$start" "$seconds" "$fin" "$fout" "$gain")" \
        -c:a aac -b:a 96k -ar 48000 -ac 1 -map_metadata -1 -map_chapters -1 -fflags +bitexact -flags:a +bitexact \
        -movflags +faststart -f ipod "$tmpclip"
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "FAILED  $id: ffmpeg exit $rc while cutting $src. Nothing was written."
        rm -f "$tmpclip"; sfailed=$((sfailed + 1)); continue
      fi
      why=$(sound_differs "$tmpclip" "$seconds" "$target")
      if [ -n "$why" ]; then
        echo "FAILED  $id: the new clip $why. Nothing was written."
        rm -f "$tmpclip"; sfailed=$((sfailed + 1)); continue
      fi
      chmod 644 "$tmpclip"
      mv -f "$tmpclip" "$clip"
      got=$(sha_of "$clip")
      if [ $same = 1 ] && [ "$got" != "$cut_sha" ]; then
        echo "cut     $id: $clip (sha256 $got, not the entry's $cut_sha although ffmpeg is $cut_ffmpeg: its form and loudness are checked, the bytes differ)"
      else
        echo "cut     $id: $clip (${seconds} s from ${start} s of $source_file, gain ${gain} dB, sha256 $got, ffmpeg $ffv)"
      fi
    done < <(entries)
    if [ $sfailed -ne 0 ]; then
      echo "$sfailed of $stotal effect sounds are not in place."
    elif [ $stotal -gt 0 ]; then
      echo "all $stotal effect sounds are in place."
    fi
  fi
fi

[ $failed -eq 0 ] && [ $vfailed -eq 0 ] && [ $sfailed -eq 0 ] || exit 1
exit 0
