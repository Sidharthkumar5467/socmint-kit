#!/bin/sh
# Install the repo's pre-commit secret guard as a native git hook.
set -eu
root=$(git rev-parse --show-toplevel)
cp "$root/scripts/hooks/pre-commit" "$root/.git/hooks/pre-commit"
chmod +x "$root/.git/hooks/pre-commit"
echo "installed .git/hooks/pre-commit (uses gitleaks if installed, else a built-in scan)"
