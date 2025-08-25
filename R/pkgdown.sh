#!/bin/bash
set -eu

cat <<EOF >"${1}"
template:
  bootstrap: 5
  light-switch: true
repo:
  url:
    source: 'https://github.com/goeva-lab/found/tree/$(git rev-parse main)/R'
    home: 'https://github.com/goeva-lab/found/'
EOF
