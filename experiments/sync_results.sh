#!/bin/bash
# Compress finished run records so they can be versioned (raw JSON is git-ignored).
cd "$(dirname "$0")/.."
for f in results/runs/*.json; do [ -e "$f.gz" ] || gzip -9 -k "$f"; done
