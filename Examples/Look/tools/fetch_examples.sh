#!/bin/bash
# fetch_examples.sh - puts the three example photos of Examples/Look into Examples/Look/Examples/, where the app
# bundles them from.
#
# The repository holds no images, so the photos are kept by reference: each entry of tools/examples.json (next to
# this script) names the URL its bytes come from, their size and their sha256. This script downloads each photo with
# curl, checks both numbers, and writes it under the file name the app reads. A file already in place with the right
# size and sha256 is left alone.
#
#   tools/fetch_examples.sh                  fetch what is missing into Examples/Look/Examples/, then check all three
#   tools/fetch_examples.sh --check          only check the files already there (no network)
#   tools/fetch_examples.sh [--check] DIR    the same with DIR as the destination
#
# Needs bash, curl and shasum only (all three ship with macOS). Exit status 0 when all three are in place and match,
# 1 otherwise (the reason per photo is printed), 2 on a usage error or an unreadable examples.json. The photos are
# CC0 1.0 (examples.json, README: Example photos). When a URL stops serving these bytes, the script says so and names
# the photo's Openverse record and source page, where the photo can be found again.

set -u

here=$(cd "$(dirname "$0")" && pwd)
manifest=$here/examples.json
user_agent="coreai-kit-look-examples/1 (+https://github.com/john-rocky/coreai-kit)"

check_only=0
dest=""
for arg in "$@"; do
  case $arg in
    --check) check_only=1 ;;
    -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
    *)
      if [ -n "$dest" ]; then echo "one destination only (see --help)" >&2; exit 2; fi
      dest=$arg ;;
  esac
done
[ -n "$dest" ] || dest=$(cd "$here/.." && pwd)/Examples

if [ ! -r "$manifest" ]; then
  echo "cannot read $manifest" >&2
  exit 2
fi

# One line per photo, read from examples.json as it is written (one "key": value pair per line), fields separated by
# the unit separator \037 (a tab would let an empty field shift the rest): id, file, url, bytes, sha256,
# openverse_record, source_page.
entries() {
  awk '
    /^ *"(id|file|url|bytes|sha256|openverse_record|source_page)": / {
      line = $0
      sub(/^ *"/, "", line)
      key = line; sub(/".*/, "", key)
      val = line; sub(/^[^:]*: */, "", val); sub(/,? *$/, "", val); gsub(/^"|"$/, "", val)
      rec[key] = val
      next
    }
    /^ *},? *$/ {
      if ("id" in rec)
        print rec["id"] "\037" rec["file"] "\037" rec["url"] "\037" rec["bytes"] "\037" rec["sha256"] "\037" rec["openverse_record"] "\037" rec["source_page"]
      split("", rec)
    }
  ' "$manifest"
}

size_of() { wc -c < "$1" | tr -d ' '; }
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }

total=0
failed=0
while IFS=$'\037' read -r id file url bytes sha record page; do
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
  if ! mkdir -p "$dest"; then
    echo "FAILED  $id: cannot create $dest"
    failed=$((failed + 1))
    continue
  fi
  tmp=$(mktemp "$dest/.fetch-XXXXXX") || { echo "FAILED  $id: cannot write in $dest"; failed=$((failed + 1)); continue; }
  curl -fsSL --retry 2 --connect-timeout 20 --max-time 120 -A "$user_agent" -o "$tmp" "$url" < /dev/null
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAILED  $id: could not download $url (curl exit $rc). Nothing was written."
    rm -f "$tmp"
    failed=$((failed + 1))
    continue
  fi
  got_bytes=$(size_of "$tmp")
  got_sha=$(sha_of "$tmp")
  if [ "$got_bytes" != "$bytes" ] || [ "$got_sha" != "$sha" ]; then
    echo "FAILED  $id: $url sent $got_bytes bytes, sha256 $got_sha; expected $bytes bytes, sha256 $sha." \
      "The source no longer serves the same file: find the photo again from $record or $page. Nothing was written."
    rm -f "$tmp"
    failed=$((failed + 1))
    continue
  fi
  chmod 644 "$tmp"
  mv -f "$tmp" "$path"
  echo "fetched $id: $path ($got_bytes bytes, sha256 ${sha:0:12}...) from $url"
done < <(entries)

if [ $total -eq 0 ]; then
  echo "no photos found in $manifest" >&2
  exit 2
fi
if [ $failed -ne 0 ]; then
  echo "$failed of $total example photos are not in place."
  exit 1
fi
echo "all $total example photos are in place."
exit 0
