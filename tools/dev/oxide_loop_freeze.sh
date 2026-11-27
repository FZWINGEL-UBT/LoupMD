#!/usr/bin/env bash
# Run in Windows Git Bash once, from the Stage-0 checkout. Never archives the dirty working tree.
# Output belongs to the WSL campaign, not to either development checkout.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${OXIDE_LOOP_WSL_CAMPAIGN:=//wsl.localhost/Ubuntu/home/fzwin/loup-s-stage0/sto320}"
root=$OXIDE_LOOP_WSL_CAMPAIGN/oxide-loop-v1
mkdir -p "$root"
check() {
    (cd "$1" && printf '%s  %s\n' \
        7461141c9323e25460935f76bbc53a2039c131f7778c9e7a3d95ae5d35ccf05b train.py \
        3d1edd3299bd27f169dedf4ee8b4a0078edf25f69fd76afa0e8393d970a41caf colab_train.sh | sha256sum -c -)
}
if [[ -d $root/stock-trainer ]]; then check "$root/stock-trainer"; exit; fi
tmp=$(mktemp -d "$root/.stock-trainer.XXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
git archive a095cf6b7b7e6109d2ba081d536d1c4dd5debf42 loupmd/distill/train.py tools/dev/colab_train.sh |
    tar -x --strip-components=2 -C "$tmp"
check "$tmp"
mv -T "$tmp" "$root/stock-trainer"
