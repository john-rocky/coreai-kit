#!/bin/bash
# fetch_examples.sh - puts the three example photos of Examples/Cutout into Examples/Cutout/Examples/, where the app
# bundles them from.
#
# The repository holds no images, so the examples are kept by reference: each entry of tools/examples.json (next to this
# script) names the URL its bytes come from, their size and their sha256. This script downloads each photo with curl,
# checks both numbers, and writes it under the file name the app reads. A file already in place with the right size and
# sha256 is left alone.
#
#   tools/fetch_examples.sh            fetch what is missing, then check everything
#   tools/fetch_examples.sh --check    only check the files already there (no network)
#   tools/fetch_examples.sh [--check] [DIR]    DIR for the photos (default Examples/Cutout/Examples)
#
# Needs bash, curl and shasum only (all three ship with macOS). Exit status 0 when every photo is in place and matches,
# 1 otherwise (the reason per file is printed), 2 on a usage error or an unreadable examples.json. Two photos are CC0 1.0
# and one is a NASA photo in the public domain (examples.json). When a URL stops serving these bytes, the script says so
# and names where the photo can be found again.

set -u

here=$(cd "$(dirname "$0")" && pwd)
manifest=$here/examples.json
user_agent="coreai-kit-cutout-examples/1 (+https://github.com/john-rocky/coreai-kit)"

check_only=0
dest=""
while [ $# -gt 0 ]; do
  case $1 in
    --check) check_only=1 ;;
    -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    *)
      if [ -n "$dest" ]; then echo "one destination only (see --help)" >&2; exit 2; fi
      dest=$1 ;;
  esac
  shift
done
[ -n "$dest" ] || dest=$(cd "$here/.." && pwd)/Examples

if [ ! -r "$manifest" ]; then
  echo "cannot read $manifest" >&2
  exit 2
fi

# One line per entry, read from examples.json as it is written (one "key": value pair per line), fields separated by
# the unit separator \037: id, file, url, bytes, sha256, source_page.
entries() {
  awk '
    /^ *"(id|file|url|bytes|sha256|source_page)": / {
      line = $0
      sub(/^ *"/, "", line)
      key = line; sub(/".*/, "", key)
      val = line; sub(/^[^:]*: */, "", val); sub(/,? *$/, "", val); gsub(/^"|"$/, "", val)
      rec[key] = val
      next
    }
    /^ *},? *$/ {
      if ("id" in rec)
        print rec["id"] "\037" rec["file"] "\037" rec["url"] "\037" rec["bytes"] "\037" rec["sha256"] "\037" rec["source_page"]
      split("", rec)
    }
  ' "$manifest"
}

size_of() { wc -c < "$1" | tr -d ' '; }
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }

total=0
failed=0
mkdir -p "$dest" || { echo "cannot create $dest" >&2; exit 1; }
while IFS=$'\037' read -r id file url bytes sha page; do
  total=$((total + 1))
  if [ -z "$file" ] || [ -z "$url" ] || [ -z "$bytes" ] || [ ${#sha} -ne 64 ]; then
    echo "FAILED  ${id:-?}: examples.json lacks file, url, bytes or sha256 for this photo"
    failed=$((failed + 1)); continue
  fi
  path=$dest/$file
  if [ -f "$path" ] && [ "$(size_of "$path")" = "$bytes" ] && [ "$(sha_of "$path")" = "$sha" ]; then
    echo "ok      $id: $path ($bytes bytes, sha256 ${sha:0:12}...) already in place"
    continue
  fi
  if [ $check_only = 1 ]; then
    echo "MISSING $id: $path (run without --check to fetch it)"
    failed=$((failed + 1)); continue
  fi
  tmp=$(mktemp "$dest/.fetch-XXXXXX") || { echo "FAILED  $id: cannot write in $dest"; failed=$((failed + 1)); continue; }
  if ! curl -fsSL --retry 2 --connect-timeout 20 --max-time 120 -A "$user_agent" -o "$tmp" "$url" < /dev/null; then
    echo "FAILED  $id: could not download $url. Nothing was written. Find the photo again from $page."
    rm -f "$tmp"; failed=$((failed + 1)); continue
  fi
  got_bytes=$(size_of "$tmp"); got_sha=$(sha_of "$tmp")
  if [ "$got_bytes" != "$bytes" ] || [ "$got_sha" != "$sha" ]; then
    echo "FAILED  $id: $url sent $got_bytes bytes, sha256 $got_sha; expected $bytes, $sha. Nothing was written. See $page."
    rm -f "$tmp"; failed=$((failed + 1)); continue
  fi
  chmod 644 "$tmp"
  mv -f "$tmp" "$path"
  echo "fetched $id: $path ($got_bytes bytes, sha256 ${got_sha:0:12}...)"
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
