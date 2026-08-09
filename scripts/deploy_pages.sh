#!/usr/bin/env bash
# Publish site/ to the gh-pages branch.
#
# Pushing a branch called gh-pages whose root is the site auto-enables GitHub
# Pages. This is done by hand rather than by an Action because the publishing
# token here was not granted GitHub's `workflow` scope, so any push touching
# .github/workflows/** is rejected outright. The workflow that would have done
# this is parked at deploy/github-pages-workflow.yml for anyone whose token can
# install it.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [[ ! -f site/data/summary.json ]]; then
  echo "site/data/summary.json is missing — run voiceval.report first" >&2
  exit 1
fi

REMOTE="$(git remote get-url origin)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cp -r site/. "$TMP/"
# Jekyll would otherwise ignore any path beginning with an underscore.
touch "$TMP/.nojekyll"

cd "$TMP"
git init -q
git checkout -q -b gh-pages
git add -A
git -c user.email="carlos.rymer@gmail.com" -c user.name="Carlos Rymer" \
    commit -q -m "Publish results site"
git remote add origin "$REMOTE"
git push -q --force origin gh-pages

echo "Pushed gh-pages. Pages will serve it within a minute or two."
